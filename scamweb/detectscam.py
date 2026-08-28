from urllib.parse import urlparse

import numpy as np


def detect_isscam(x: str, sess):
    ## whitelist for known domains
    whitelist = {
        "facebook.com",
        "twitter.com",
        "google.com",
        "youtube.com",
        "bbc.com",
        "linkedin.com",
        "instagram.com",
        "reddit.com",
        "wikipedia.org",
        "amazon.com",
        "github.com",
        "apple.com",
        "microsoft.com",
        "yahoo.com",
        "whatsapp.com",
        "pinterest.com",
        "vimeo.com",
        "dropbox.com",
        "nytimes.com",
        "theguardian.com",
        "wordpress.com",
        "bbc.co.uk",
        "cnn.com",
        "aljazeera.com",
        "forbes.com",
        "bbc.co.uk",
        "zoom.us",
        "twitch.tv",
        "skype.com",
        "snapchat.com",
        "telegram.org",
        "messenger.com",
        "slack.com",
        "airbnb.com",
        "uber.com",
        "spotify.com",
        "soundcloud.com",
        "quora.com",
        "github.io",
        "trello.com",
        "notion.so",
        "merriam-webster.com",
        "thedailybeast.com",
        "cnbc.com",
        "huffpost.com",
        "businessinsider.com",
        "thesaurus.com",
        "time.com",
        "fortune.com",
        "wired.com",
        "lifehacker.com",
        "techcrunch.com",
        "digitaltrends.com",
        "engadget.com",
        "xfinity.com",
        "samsung.com",
        "hulu.com",
        "adobe.com",
        "wellsfargo.com",
        "paypal.com",
        "squareup.com",
        "expedia.com",
        "booking.com",
        "bestbuy.com",
        "target.com",
        "lowes.com",
        "homeDepot.com",
        "newegg.com",
        "ebay.com",
        "shopify.com",
        "etsy.com",
    }
    parsed_url = urlparse(x)

    domain = parsed_url.netloc.lower()

    for trusted_domain in whitelist:
        if trusted_domain in domain:
            return False
    a = len(x)
    b = 1 if "subscribe" in x else 0
    c = 1 if "#" in x else 0
    d = sum(1 for i in x if i.isnumeric())
    e = 0 if "https" in x else 1
    f = len(x.split("/"))
    ##features preparation
    input_dict = {
        "len_url": np.array([[a]], dtype=np.float32),
        "contains_subscribe": np.array([[b]], dtype=np.float32),
        "contains_hash": np.array([[c]], dtype=np.float32),
        "num_digits": np.array([[d]], dtype=np.float32),
        "non_https": np.array([[e]], dtype=np.float32),
        "num_words": np.array([[f]], dtype=np.float32),
    }

    # Predict
    pred_ort = sess.run(None, input_dict)
    
    # Safely extract probability supporting various ONNX output formats
    if len(pred_ort) > 1:
        output = pred_ort[1]
    else:
        output = pred_ort[0]

    if isinstance(output, list) and len(output) > 0 and isinstance(output[0], dict):
        prob = output[0].get(1, output[0].get('1', 0.0))
    else:
        row = output[0]
        prob = row[1] if len(row) > 1 else row[0]

    # return true if prob is greater than 85%
    return prob >= 0.82
