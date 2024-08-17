import os

import functions_framework
from cloudevents.http import CloudEvent
from flask import jsonify

project_id = os.environ["GCP_PROJECT"]


@functions_framework.cloud_event
def temp_consumer_pubsub(cloud_event: CloudEvent):
    """Background Cloud Function.
    Args:
        cloudevent (dict): The dictionary with data specific to this type of
                           event. The `data` field contains the Pub/Sub
                           message. The `attributes` field will contain custom
                           attributes if there are any.
    """
    # Triggered from a message on a Cloud Pub/Sub topic.

    expected_type = "google.cloud.pubsub.topic.v1.messagePublished"
    received_type = cloud_event["type"]
    # 위 어디서 자신의 topic을 갖는지 알지?
    if received_type != expected_type:
        raise ValueError(
            f"Expected event type {expected_type} but received {received_type}"
        )
        
    print(cloud_event)

    data = cloud_event.data['message']['data']
    print(data)
    return jsonify({"status": "ok"})

