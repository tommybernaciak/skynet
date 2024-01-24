import openai
openai.api_key = "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"
response = openai.File.create(
  file=open("data.jsonl", "rb"),
  purpose='fine-tune'
)

file_id = response['id']
print("file id:")
print(file_id)

jobResponse = openai.FineTuningJob.create(training_file=file_id, model="gpt-3.5-turbo")
job_id = jobResponse['id']

print("job id:")
print(job_id)
