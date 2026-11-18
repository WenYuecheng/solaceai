package com.mindbridge.agent.runtime;

import java.nio.channels.*;
import java.nio.file.*;
import java.security.MessageDigest;
import java.nio.charset.StandardCharsets;
import java.util.*;

/** Nonblocking cross-process file leases released by the operating system on process exit. */
public final class TurnLocks {
    private final Path root;
    public TurnLocks(Path root){this.root=root;try{Files.createDirectories(root);}catch(Exception e){throw new IllegalStateException("Cannot create turn lock directory",e);}}
    public Lease acquire(long user,String session,String run){List<FileChannel> channels=new ArrayList<>();List<FileLock> locks=new ArrayList<>();try{
        List<String> keys=new ArrayList<>();if(session!=null&&!session.isBlank())keys.add("session:"+user+":"+session);keys.add("run:"+user+":"+run);
        for(String key:keys){FileChannel channel=FileChannel.open(root.resolve(hash(key)+".lock"),StandardOpenOption.CREATE,StandardOpenOption.WRITE);channels.add(channel);FileLock lock=channel.tryLock();if(lock==null)throw new IllegalStateException("本次对话正在回复，请等待完成。");locks.add(lock);}return new Lease(channels,locks);
    }catch(Exception e){new Lease(channels,locks).close();throw new IllegalStateException("本次对话正在回复，请等待完成。",e);}}
    public static String hash(String value){try{return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(value.getBytes(StandardCharsets.UTF_8)));}catch(Exception e){throw new IllegalStateException(e);}}
    public static final class Lease implements AutoCloseable{private final List<FileChannel> channels;private final List<FileLock> locks;private Lease(List<FileChannel> c,List<FileLock> l){channels=c;locks=l;}public void close(){for(FileLock l:locks)try{if(l.isValid())l.release();}catch(Exception ignored){}for(FileChannel c:channels)try{c.close();}catch(Exception ignored){}}}
}
