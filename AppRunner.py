import tweepy
import csv
CK = 'yNvvRgMYOrYTA1j0vGwtjgT8Y'   # Consumer Key
CS = '5u7sqseTJ4vJxG8CNp1x9lASyz3j4KkYmX2bheFwojwWvfyl9P'   # Consumer Secret
AT = '3086156577-VDqASN5kNizAx9QMxbPLPWnNsCqdTgS77XuqQSv'   # Access Token
AS = 'HZIzTUTGADbTc0rkS1evkk2zZhrZFaMRfB32yqLT31kNe'   # Accesss Token Secert
auth = tweepy.auth.OAuthHandler(CK, CS)
auth.set_access_token(AT, AS)
api = tweepy.API(auth)

csvFile = open('result.csv', 'a')
csvWriter = csv.writer(csvFile)

for tweet in tweepy.Cursor(api.user_timeline,id ="hurricane florence",
                           # q="cancer",
                           count=100,
                           result_type="recent",
                           lang="en").items():
    #print tweet.text
    csvWriter.writerow([tweet.text.encode('utf-8')])
    print tweet.text #tweet.created_at,

    (last_id, tweet, user_name) = (tweet.id_str, tweet.text, tweet.user.screen_name)
csvFile.close()
#current_id=last_id