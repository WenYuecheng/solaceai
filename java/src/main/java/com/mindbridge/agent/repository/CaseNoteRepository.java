package com.mindbridge.agent.repository;
import com.mindbridge.agent.domain.CaseNote;
import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;
public interface CaseNoteRepository extends JpaRepository<CaseNote,Long> {List<CaseNote> findByCaseIdOrderByCreatedAtAscIdAsc(long caseId);}
