UNWIND $comments AS comment
WITH comment
WHERE comment.id IS NOT NULL AND comment.redditId IS NOT NULL
MATCH (r:Reddit {redditId:comment.redditId})
MATCH (c:Comment {commentId:comment.id})
MERGE (r)-[:HAS_COMMENT]->(c);
