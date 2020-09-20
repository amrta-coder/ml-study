
import json
import os.path

from hbconfig import Config
import requests


def send_message_to_slack(config_name):
    project_name = os.path.basename(os.path.abspath("."))

    data = {
        "text": f"トレニングが完了しました： *{project_name}* ！　利用している設定ファイル： `{config_name}`."
    }

    webhook_url = Config.slack.webhook_url
    if webhook_url == "":
        print(data["text"])
    else:
        requests.post(Config.slack.webhook_url, data=json.dumps(data))
