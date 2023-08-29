CREATE CONSTRAINT subredditConstraint IF NOT EXISTS
FOR (n:Subreddit)
REQUIRE n.subredditId IS UNIQUE
