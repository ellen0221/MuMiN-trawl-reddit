UNWIND $reddits AS reddit
WITH reddit
WHERE reddit.id IS NOT NULL and reddit.author_fullname IS NOT NULL
MATCH (r:Reddit {redditId:reddit.id})
MATCH (u:User {userFullname:reddit.author_fullname})
MERGE (u)-[:POSTED]->(r);
