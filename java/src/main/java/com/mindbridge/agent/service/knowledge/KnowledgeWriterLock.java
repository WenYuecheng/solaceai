package com.mindbridge.agent.service.knowledge;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.file.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantLock;

/** Advisory writer lock. Every process using a database must share this persistent path. */
final class KnowledgeWriterLock {
    private static final ConcurrentHashMap<Path,ReentrantLock> LOCAL=new ConcurrentHashMap<>();
    private final Path path;
    KnowledgeWriterLock(Path path) { this.path=path.toAbsolutePath().normalize(); }
    Guard acquire() {
        ReentrantLock local=LOCAL.computeIfAbsent(path,p->new ReentrantLock());
        local.lock(); FileChannel channel=null;
        try {
            Files.createDirectories(path.getParent());
            channel=FileChannel.open(path,StandardOpenOption.CREATE,StandardOpenOption.WRITE);
            return new Guard(local,channel,channel.lock());
        } catch(IOException | RuntimeException e) {
            if(channel!=null) try {channel.close();} catch(IOException ignored) {}
            local.unlock(); throw new IllegalStateException("Cannot acquire knowledge writer lock",e);
        }
    }
    static final class Guard implements AutoCloseable {
        private final ReentrantLock local; private final FileChannel channel; private final FileLock file;
        private boolean closed;
        Guard(ReentrantLock local,FileChannel channel,FileLock file) {this.local=local;this.channel=channel;this.file=file;}
        public void close() {
            if(closed) return; closed=true;
            try {file.release();} catch(IOException e) {throw new IllegalStateException("Cannot release knowledge writer lock",e);}
            finally {try {channel.close();} catch(IOException ignored) {} finally {local.unlock();}}
        }
    }
}
