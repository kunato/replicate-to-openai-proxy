import os
os.environ['OPENAI_API_BASE'] = 'http://localhost/v1'
import openai
openai.api_key = "none"
REPLICATE_MODEL_PATH = "replicate/vicuna-13b:e6d469c2b11008bb0e446c3e9629232f9674581224536851272c54871f84076e"

def main():
    response = openai.Completion.create(model=REPLICATE_MODEL_PATH, 
                                        prompt="Say this is a test", 
                                        temperature=1, max_tokens=100, stream=True)
    print('response')
    for resp in response:
        print(resp.choices[0].text)

if __name__ == '__main__':
    main()