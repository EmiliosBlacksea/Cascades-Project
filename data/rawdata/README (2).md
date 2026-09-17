** Weibo Dataset **


119,313 messages in June 1, 2016. Each line contains the information of a certain message, the format of which is:

<message_id>\tab<user_id>\tab<publish_time>\tab<retweet_number>\tab<retweets>
<message_id>:     the unique id of each message, ranging from 1 to 119,313.
<root_user_id>:   the unique id of root user. The user id ranges from 1 to 6,738,040.
<publish_time>:   the publish time of this message, recorded as unix timestamp.
<retweet_number>: the total number of retweets of this message within 24 hours.
<retweets>:       the retweets of this message, each retweet is split by " ". Within each retweet, it records 
the entile path for this retweet, the format of which is <user1>/<user2>/......<user n>:<retweet_time>.