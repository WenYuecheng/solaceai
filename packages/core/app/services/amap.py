"""Allowlisted official Amap MCP hospital lookup; never synthesizes fallback POIs."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from functools import lru_cache
import json
import logging
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

CATEGORIES = ('精神卫生中心', '医院心理科', '急诊')
NOTICE = '高德地图地点信息仅供寻址参考；医院科室、接诊时间和挂号方式请向院方核实。紧急危险优先联系当地急救及能到场的人，不等待地图查询。'


@lru_cache(maxsize=1)
def city_aliases():
    return json.loads((Path(__file__).resolve().parents[1] / 'assets' / 'amap-cities.json').read_text())['cities']


def normalize_city(city: str) -> str:
    normalized = city_aliases().get(city.strip())
    if not normalized:
        raise ValueError('请提供明确的中国城市名称')
    return normalized


class _RedactKey(logging.Filter):
    def filter(self, record):
        record.msg = re.sub(r'([?&]key=)[^\s\"\'&]+', r'\1[REDACTED]', record.getMessage())
        record.args = ()
        # HTTPStatusError and SDK exception groups can embed the keyed URL.
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return True


# The official MCP endpoint authenticates in the URL; strip it from HTTP logs.
for _logger_name in ('httpx', 'httpx2', 'mcp.client.streamable_http'):
    logging.getLogger(_logger_name).addFilter(_RedactKey())


class AmapHospitalService:
    def __init__(self, settings):
        self.settings = settings

    @contextmanager
    def _db(self, deadline):
        path = Path(self.settings.amap_usage_path)
        if not path.is_absolute():
            path = self.settings.project_root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path, timeout=max(.001, min(.1, deadline-time.monotonic())))
        db.execute('CREATE TABLE IF NOT EXISTS calls (day TEXT PRIMARY KEY, count INTEGER NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS cache (query TEXT PRIMARY KEY, expires REAL, payload TEXT)')
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def search(self, city: str, category: str, limit: int = 3) -> dict:
        deadline = time.monotonic() + self.settings.amap_timeout_seconds
        city = normalize_city(city)
        if not re.fullmatch(r'[\u4e00-\u9fffA-Za-z ·-]{2,40}', city) or category not in CATEGORIES or not 1 <= limit <= 5:
            raise ValueError('请提供明确城市和有效的医院类别')
        empty = dict(status='unavailable', source='高德地图 MCP', hospitals=[], notice=NOTICE)
        if not self.settings.amap_enabled or not self.settings.amap_api_key.strip():
            return empty
        query = json.dumps([city, category], ensure_ascii=False)
        with self._db(deadline) as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('DELETE FROM cache WHERE expires <= ?', (time.time(),))
            cached = db.execute('SELECT payload FROM cache WHERE query=?', (query,)).fetchone()
            if cached:
                result = json.loads(cached[0]); result['hospitals'] = result['hospitals'][:limit]
                return result
            day = datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
            db.execute('INSERT OR IGNORE INTO calls VALUES (?,0)', (day,))
            changed = db.execute('UPDATE calls SET count=count+1 WHERE day=? AND count < ?', (day, self.settings.amap_daily_call_limit))
            if changed.rowcount != 1:
                return empty
        try:
            async def bounded_query():
                async with asyncio.timeout(max(.001, deadline-time.monotonic())):
                    return await self._remote(city, category)
            payload = asyncio.run(bounded_query())
            pois = payload.get('pois')
            if not isinstance(pois, list):
                raise ValueError('Malformed provider response')
            hospitals = []
            for poi in pois[:30]:
                if not isinstance(poi, dict) or not any(code.startswith(('0901','0902')) for code in str(poi.get('typecode', '')).split('|')):
                    continue
                name, address = poi.get('name'), poi.get('address')
                if not isinstance(name, str) or not name.strip() or not isinstance(address, str) or not address.strip():
                    continue
                name, address = name[:120], address[:240]
                hospitals.append(dict(name=name, address=address, city=city, map_url='https://uri.amap.com/search?' + urlencode({'keyword': name, 'city': city})))
                if len(hospitals) == 5: break
            result = dict(status='ok' if hospitals else 'no_results', source='高德地图 MCP', hospitals=hospitals, notice=NOTICE)
        except Exception:
            # Never retain exception URLs, response text, or provider credentials.
            raise RuntimeError('医院信息暂时无法核实，请稍后重试或向当地医院核实') from None
        if time.monotonic() >= deadline:
            raise RuntimeError('医院信息暂时无法核实，请稍后重试或向当地医院核实')
        with self._db(deadline) as db:
            db.execute('INSERT OR REPLACE INTO cache VALUES (?,?,?)', (query, time.time()+self.settings.amap_cache_ttl_seconds, json.dumps(result, ensure_ascii=False)))
        return {**result, 'hospitals': hospitals[:limit]}

    async def _remote(self, city: str, category: str) -> dict:
        async with asyncio.timeout(self.settings.amap_timeout_seconds):
            # Fixed endpoint and fixed tool. No arbitrary URL/tool from model input.
            async with streamable_http_client('https://mcp.amap.com/mcp?' + urlencode({'key': self.settings.amap_api_key}), terminate_on_close=False) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    result = await session.call_tool('maps_text_search', {'keywords': category, 'city': city, 'citylimit': True})
                    if result.is_error:
                        raise ValueError('Provider query failed')
                    for item in result.content:
                        if hasattr(item, 'text'):
                            parsed = json.loads(item.text)
                            if isinstance(parsed, dict) and isinstance(parsed.get('pois'), list):
                                return parsed
                    raise ValueError('Missing POI response')
