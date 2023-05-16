import json
import openai

#Replace with real key
api_key ="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"
openai.api_key = api_key

file_name = r"C:\Users\elelm\Downloads\training3.jsonl"

prepare_training=True

#Prepare data for training
if prepare_training:
    upload_response = openai.File.create(
    file=open(file_name, "rb"),
    purpose='fine-tune'
    )
    file_id = upload_response.id
    upload_response
    print("Upload response:")
    print(upload_response)
    file_id = upload_response["id"]

else:
    #Use prepared data for training.
    file_id = "file-s615XMkyr8xNHZHI8nA53wVF" #With subjects
    #file_id = "file-vj5Wq7YSdnfQCf5LUWoyZ5wA" #Without subjects

#Choose model to train
fine_tune_response = openai.FineTune.create(training_file=file_id, model="ada")
print(fine_tune_response)
