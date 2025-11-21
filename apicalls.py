import requests
import json

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"


# Call each API endpoint and store the responses

# Call prediction endpoint
response1 = requests.post(
    f"{URL}/prediction", json={"filepath": "testdata/testdata.csv"}
).text

# Call scoring endpoint
response2 = requests.get(f"{URL}/scoring").text

# Call summary stats endpoint
response3 = requests.get(f"{URL}/summarystats").text

# Call diagnostics endpoint
response4 = requests.get(f"{URL}/diagnostics").text

# combine all API responses
responses = {
    "prediction": response1,
    "scoring": response2,
    "summarystats": response3,
    "diagnostics": response4,
}  # combine reponses here

# write the responses to your workspace

# Write responses to file
with open("apireturns.txt", "w") as f:
    f.write(json.dumps(responses, indent=4))

print("API responses saved to apireturns.txt")
