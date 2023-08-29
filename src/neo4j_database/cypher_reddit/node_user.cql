UNWIND $reddits AS reddit
WITH reddit
WHERE reddit.author_fullname IS NOT NULL
MERGE (n:User {userFullname:reddit.author_fullname})
