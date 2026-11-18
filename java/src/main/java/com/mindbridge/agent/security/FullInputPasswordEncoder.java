package com.mindbridge.agent.security;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Base64;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;

/** Versioned UTF-8 SHA-256 prehash avoids BCrypt's 72-byte input truncation. */
public final class FullInputPasswordEncoder implements PasswordEncoder {
    private static final String PREFIX = "{bcrypt-sha256-v1}";
    private final BCryptPasswordEncoder bcrypt;

    public FullInputPasswordEncoder() { this(10); }
    public FullInputPasswordEncoder(int strength) { bcrypt = new BCryptPasswordEncoder(strength); }

    @Override
    public String encode(CharSequence rawPassword) {
        return PREFIX + bcrypt.encode(prehash(rawPassword));
    }

    @Override
    public boolean matches(CharSequence rawPassword, String encodedPassword) {
        if (rawPassword == null || encodedPassword == null) return false;
        if (encodedPassword.startsWith(PREFIX)) {
            return bcrypt.matches(prehash(rawPassword), encodedPassword.substring(PREFIX.length()));
        }
        // Existing Java accounts remain readable; all newly written hashes use v1.
        return encodedPassword.matches("\\$2[aby]\\$.*") && bcrypt.matches(rawPassword, encodedPassword);
    }

    @Override
    public boolean upgradeEncoding(String encodedPassword) {
        return encodedPassword == null || !encodedPassword.startsWith(PREFIX)
                || bcrypt.upgradeEncoding(encodedPassword.substring(PREFIX.length()));
    }

    private String prehash(CharSequence password) {
        try {
            return Base64.getEncoder().encodeToString(MessageDigest.getInstance("SHA-256")
                    .digest(password.toString().getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException(impossible);
        }
    }
}
