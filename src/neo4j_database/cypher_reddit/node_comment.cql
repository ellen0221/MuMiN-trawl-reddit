UNWIND $comments AS comment
WITH comment
WHERE comment.id IS NOT NULL
MERGE (n:Comment {commentId:comment.id})
ON CREATE SET
    n.text = comment.body,
    n.name = comment.name,
    n.createdAt = comment.created_utc,
    n.ups = toInteger(comment.score);
