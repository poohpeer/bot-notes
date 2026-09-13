"""Pure logic: ACL predicates, chunking, ranking.

No aiogram, RQ, or SQLAlchemy session imports — only types. That is what
makes ACL predicates and chunking testable without infrastructure; see
docs/architecture/01-context.md.
"""
