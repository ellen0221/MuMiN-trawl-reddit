UNWIND $reddits as reddit
WITH reddit
WHERE reddit.post_hint = 'link'
MERGE (:Url {name:reddit.url});
