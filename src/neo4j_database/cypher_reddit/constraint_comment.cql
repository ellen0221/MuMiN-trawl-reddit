CREATE CONSTRAINT commentConstraint IF NOT EXISTS
FOR (n:Comment)
REQUIRE n.commentId IS UNIQUE
