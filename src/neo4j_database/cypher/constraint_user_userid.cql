CREATE CONSTRAINT userIdConstraint IF NOT EXISTS ON (n:User)
ASSERT n.userId IS UNIQUE;
