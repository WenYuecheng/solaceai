package com.mindbridge.agent.config;

import com.mindbridge.agent.domain.UserAccount;
import com.mindbridge.agent.repository.UserAccountRepository;
import com.mindbridge.agent.service.knowledge.KnowledgeIngestionService;
import java.util.Set;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;

@Component
@Order(Ordered.HIGHEST_PRECEDENCE)
public class DataInitializer implements ApplicationRunner {

    private final UserAccountRepository userAccountRepository;
    private final PasswordEncoder passwordEncoder;
    private final KnowledgeIngestionService knowledgeIngestionService;
    private final boolean demoUsers;
    private final boolean bundledKnowledge;
    private final String initialAdminPassword;

    public DataInitializer(
            UserAccountRepository userAccountRepository,
            PasswordEncoder passwordEncoder,
            KnowledgeIngestionService knowledgeIngestionService,
            @Value("${mindbridge.bootstrap.demo-users:true}") boolean demoUsers,
            @Value("${mindbridge.bootstrap.admin-password:}") String initialAdminPassword,
            @Value("${mindbridge.bootstrap.bundled-knowledge:true}") boolean bundledKnowledge
    ) {
        this.userAccountRepository = userAccountRepository;
        this.passwordEncoder = passwordEncoder;
        this.knowledgeIngestionService = knowledgeIngestionService;
        this.demoUsers = demoUsers;
        this.bundledKnowledge = bundledKnowledge;
        this.initialAdminPassword = initialAdminPassword;
    }

    @Override
    public void run(ApplicationArguments args) {
        seedUsers();
        if (bundledKnowledge) {
            knowledgeIngestionService.syncClasspathKnowledge();
        }
    }

    private void seedUsers() {
        if (userAccountRepository.count() > 0) {
            return;
        }
        if (!demoUsers && initialAdminPassword.isBlank()) {
            return;
        }
        if (!initialAdminPassword.isEmpty() && (initialAdminPassword.isBlank()
                || initialAdminPassword.length() < 16 || initialAdminPassword.length() > 128)) {
            throw new IllegalStateException("Bootstrap admin password must be 16 to 128 characters");
        }
        UserAccount admin = new UserAccount();
        admin.setUsername("admin");
        admin.setDisplayName("Counselor Admin");
        admin.setPassword(passwordEncoder.encode(initialAdminPassword.isEmpty() ? "admin123" : initialAdminPassword));
        admin.setRoles(Set.of("ROLE_ADMIN", "ROLE_USER"));
        userAccountRepository.save(admin);

        if (!demoUsers) {
            return;
        }
        UserAccount student = new UserAccount();
        student.setUsername("student");
        student.setDisplayName("Demo Student");
        student.setPassword(passwordEncoder.encode("student123"));
        student.setRoles(Set.of("ROLE_USER"));
        userAccountRepository.save(student);
    }
}
