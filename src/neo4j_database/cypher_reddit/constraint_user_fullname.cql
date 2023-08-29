CREATE CONSTRAINT userFullnameConstraint IF NOT EXISTS
FOR (n:User)
REQUIRE n.userFullname IS UNIQUE;
