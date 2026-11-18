package com.mindbridge.agent.runtime;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.mindbridge.agent.service.PrivacySanitizer;
import java.nio.file.*;
import java.sql.*;
import java.time.Instant;
import java.util.*;
import java.util.regex.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

/** User-scoped SQLite preferences and committed rolling summaries. No model guesses are persisted. */
@Component
public class RuntimeStore {
    private final String url;
    private static final Pattern DIRECT=Pattern.compile("(?:^|[，,。！!？?；;\\n：:]|对了[，,]?)\\s*(?:请|请你|帮我|请帮我)?记住[：:\\s]*(.+)");
    private static final Pattern ANY_DIRECT=Pattern.compile("(?:请你|请帮我|帮我|请)记住[：:\\s]*(.+)");
    private static final Pattern DELETE_DIRECT=Pattern.compile("(?:请帮我|请你|帮我|请)?\\s*(?:删除(?:这条)?记忆|忘记(?:这条)?记忆)[：:#\\s]*([1-9][0-9]{0,18})[。！!]?",Pattern.UNICODE_CHARACTER_CLASS);
    @Autowired public RuntimeStore(RuntimeProperties p,ObjectMapper json){this(p.getStatePath(),json);}
    public RuntimeStore(String path,ObjectMapper json){
        try{Path target=Path.of(path).toAbsolutePath();Files.createDirectories(target.getParent());url="jdbc:sqlite:"+target;try(Connection c=connect();Statement s=c.createStatement()){
            s.execute("CREATE TABLE IF NOT EXISTS preferences(id INTEGER PRIMARY KEY AUTOINCREMENT,user_id INTEGER NOT NULL,session_id TEXT NOT NULL,content TEXT NOT NULL,created_at TEXT NOT NULL,UNIQUE(user_id,content))");
            s.execute("CREATE TABLE IF NOT EXISTS summaries(user_id INTEGER NOT NULL,session_id TEXT NOT NULL,covered INTEGER NOT NULL,content TEXT NOT NULL,PRIMARY KEY(user_id,session_id))");
        }}catch(Exception e){throw new IllegalStateException("Runtime state storage unavailable",e);}
    }
    private Connection connect() throws SQLException{Connection c=DriverManager.getConnection(url);try(Statement s=c.createStatement()){s.execute("PRAGMA busy_timeout=15000");}return c;}
    public static String explicitContent(String text){Matcher m=DIRECT.matcher(text);if(!m.find()){m=ANY_DIRECT.matcher(text);if(!m.find())return null;}String before=text.substring(0,m.end());for(String blocked:List.of("不要记住","别记住","不必记住","不用记住","引用","转述","“","「","\""))if(before.contains(blocked))return null;return m.group(1).trim().isEmpty()?null:m.group(1).trim();}
    static boolean authorizesDeletion(String text,long id){Matcher matcher=DELETE_DIRECT.matcher(text.strip());return matcher.matches()&&matcher.group(1).equals(Long.toString(id));}
    public Long remember(long user,String session,String text){String raw=explicitContent(text);if(raw==null)return null;String content=new PrivacySanitizer().sanitize(raw);if(content.codePointCount(0,content.length())>1000)content=content.substring(0,content.offsetByCodePoints(0,1000));
        try(Connection c=connect()){
            try(PreparedStatement s=c.prepareStatement("INSERT OR IGNORE INTO preferences(user_id,session_id,content,created_at) VALUES(?,?,?,?)")){s.setLong(1,user);s.setString(2,session);s.setString(3,content);s.setString(4,Instant.now().toString());s.executeUpdate();}
            try(PreparedStatement s=c.prepareStatement("SELECT id FROM preferences WHERE user_id=? AND content=?")){s.setLong(1,user);s.setString(2,content);try(ResultSet r=s.executeQuery()){return r.next()?r.getLong(1):null;}}
        }catch(SQLException e){throw new IllegalStateException("Memory save failed",e);}
    }
    public List<Map<String,Object>> memories(long user){try(Connection c=connect();PreparedStatement s=c.prepareStatement("SELECT id,session_id,content,created_at FROM preferences WHERE user_id=? ORDER BY id DESC LIMIT 100")){s.setLong(1,user);List<Map<String,Object>> out=new ArrayList<>();try(ResultSet r=s.executeQuery()){while(r.next())out.add(Map.of("id",r.getLong(1),"user_id",user,"session_id",r.getString(2),"content",r.getString(3),"created_at",r.getString(4)));}return out;}catch(SQLException e){throw new IllegalStateException("Memory read failed",e);}}
    public boolean deleteMemory(long user,long id){try(Connection c=connect();PreparedStatement s=c.prepareStatement("DELETE FROM preferences WHERE user_id=? AND id=?")){s.setLong(1,user);s.setLong(2,id);return s.executeUpdate()==1;}catch(SQLException e){throw new IllegalStateException("Memory delete failed",e);}}
    public record Summary(int covered,String content){}
    public Summary summary(long user,String session){try(Connection c=connect();PreparedStatement s=c.prepareStatement("SELECT covered,content FROM summaries WHERE user_id=? AND session_id=?")){s.setLong(1,user);s.setString(2,session);try(ResultSet r=s.executeQuery()){return r.next()?new Summary(r.getInt(1),r.getString(2)):new Summary(0,"");}}catch(SQLException e){throw new IllegalStateException("Summary read failed",e);}}
    public void saveSummary(long user,String session,Summary expected,Summary next){try(Connection c=connect()){
        c.setAutoCommit(false);try{
            try(PreparedStatement s=c.prepareStatement("INSERT OR IGNORE INTO summaries VALUES(?,?,0,'')")){s.setLong(1,user);s.setString(2,session);s.executeUpdate();}
            try(PreparedStatement s=c.prepareStatement("UPDATE summaries SET covered=?,content=? WHERE user_id=? AND session_id=? AND covered=? AND content=?")){s.setInt(1,next.covered);s.setString(2,next.content);s.setLong(3,user);s.setString(4,session);s.setInt(5,expected.covered);s.setString(6,expected.content);if(s.executeUpdate()!=1)throw new IllegalStateException("Summary changed concurrently");}c.commit();
        }catch(Exception e){c.rollback();throw e;}
    }catch(SQLException e){throw new IllegalStateException("Summary save failed",e);}}
}
