import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def text(to: str, text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "text"
    
    """
    url = f"https://ai-translation.p.rapidapi.com/translate"
    querystring = {'to': to, 'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-translation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation


if __name__ == "__main__":
    inp_text = "Hello, how are you today?"
    to = 'en'
    toolbench_rapidapi_key = "5b91ea4ff0msh05dd104cd687ca9p1bf143jsne32b1273bd84"
    observation = text(to, inp_text)
    print(observation)