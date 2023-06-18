
# FableForge: Creating Picture Books with OpenAI, Replicate, and Deep Lake

## What is FableForge?

FableForge is an app that generates picture books from a single prompt. First, GPT-3.5/4 is instructed to write a short children's book. Then, using the new function calling feature OpenAI just announced, the text from each page of the book is tranformed into a prompt for Stable Diffusion. These prompts are sent to Replicate, corresponding images are generated, and all the elements are combined for a complete picture book. The matching images and prompts are stored in a Deep Lake dataset, which allows easy analysis in the web-based UI. 

## What Did and Didn't Work while Building FableForge

Before we look at the exact solution we eventually decided on, let's take a glance at the approaches that didn't work, and what we learned from them:

### Didn't Work: Instructing the LLM To Generate Stable Diffusion Prompts

Initially, it seemed like it might be possible to just send the LLM the text of our book and tell it to generate a prompt for each page. However, this didn't work for a few reasons:

- **Stable Diffusion released in 2022**: While it might seem like Stable Diffusion is already "old news", to GPT-3.5 and GPT-4 it's in the future. Take a look at GPT-4's response to the question "What is Stable Diffusion?":
![enter image description here](https://i.imgur.com/iXeXWrR.png)

- **Teaching the LLM how to prompt is difficult**: It's possible to instruct the LLM to generate prompts without the LLM knowing what Stable Diffusion is; giving it the exact format to generate a prompt with has decent results. Unfortunately, the often injects plot details or non-visual content into the prompts, no matter how often you tell it not to. These details skew the relevance of the prompts and negatively impact the quality of the generated images.

### Did Work: Function Calling Capabilities

On June 13th, [OpenAI announced a huge update to the chat completions API - function calling!]( https://openai.com/blog/function-calling-and-other-api-updates) This means we can provide the chat model with a function, and the chat model will output a JSON object according to that function's parameters. For our Stable Diffusion prompts, we need structured data that strictly adheres to certain rules - a function is perfect for that! Let's take a look at one of the functions we used:

```json
get_visual_description_function = [{
    'name': 'get_passage_setting',
    'description': 'Generate and describe the visuals of a passage in a book. Visuals only, no characters, plot, or people.',
    'parameters': {
        'type': 'object',
        'properties': {
            'setting': {
                'type': 'string',
                'description': 'The visual setting of the passage, e.g. a green forest in the pacific northwest',
            },
            'time_of_day': {
                'type': 'string',
                'description': 'The time of day of the passage, e.g. nighttime, daytime. If unknown, leave blank.',
            },
            'weather': {
                'type': 'string',
                'description': 'The weather of the passage, eg. rain. If unknown, leave blank.',
            },
            'key_elements': {
                'type': 'string',
                'description': 'The key visual elements of the passage, eg tall trees',
            },
            'specific_details': {
                'type': 'string',
                'description': 'The specific visual details of the passage, eg moonlight',
            }
        },
        'required': ['setting', 'time_of_day', 'weather', 'key_elements', 'specific_details']
    }
}]
```

With this, we can send the chat model a page from our book, the function, and instructions telling it to infer the details from the provided page. In return we get structured data that we can use to form a great Stable Diffusion prompt!

### Perfecting the Process: Using Deep Lake for Storage and Analysis
The final step breakthrough for perfecting FableForge was using Deep Lake to store the generated images and text. With Deep Lake, we could store multiple modalities of data, such as image and text, in the cloud. The web-based UI provided by Deep Lake made it incredibly straightforward to display, analyze, and optimize the generated images and prompts, improving the quality of our picture book output. For future Stable Diffusion endeavours, we now have a huge dataset showing us what prompts work, and what prompts don't!

![enter image description here](https://i.imgur.com/or2MxIe.png)


## Conclusion: The Future of AI-Generated Picture Books with FableForge

Developing FableForge was a perfect example of how new AI tools and methodologies can be leveraged to overcome hurdles. Through leveraging the power of OpenAI's function calling feature, Stable Diffusion's image generation abilities, and Deep Lake's multimodal dataset storage and analysis capabilities, we were able to create an app that opens up a new frontier in children's picture book creation.

While we had our fair share of obstacles, from teaching the language model to generate image prompts to finding the perfect method of storing and analyzing data, every challenge brought valuable learning experiences and led to making FableForge into a working app!


