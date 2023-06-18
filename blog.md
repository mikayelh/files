# FableForge: Creating Picture Books with OpenAI, Replicate, and Deep Lake

## What is FableForge?

FableForge is an app that generates picture books from a single prompt. First, GPT-3.5/4 is instructed to write a short children's book. Then, using the new function calling feature OpenAI just announced, the text from each page of the book is tranformed into a prompt for Stable Diffusion. These prompts are sent to Replicate, corresponding images are generated, and all the elements are combined for a complete picture book. The matching images and prompts are stored in a Deep Lake dataset, which allows easy analysis in the web-based UI. 

https://github.com/e-johnstonn/FableForge/assets/30129211/9657b0de-ac80-46d1-bc30-34759b601498


## What Did and Didn't Work while Building FableForge

Before we look at the exact solution we eventually decided on, let's take a glance at the approaches that didn't work, and what we learned from them:

### Didn't Work: Instructing the LLM To Generate Stable Diffusion Prompts

Initially, it seemed like it might be possible to just send the LLM the text of our book and tell it to generate a prompt for each page. However, this didn't work for a few reasons:

- **Stable Diffusion released in 2022**: While it might seem like Stable Diffusion is already "old news", to GPT-3.5 and GPT-4 it's in the future. Take a look at GPT-4's response to the question "What is Stable Diffusion?":
![enter image description here](https://i.imgur.com/iXeXWrR.png)

- **Teaching the LLM how to prompt is difficult**: It's possible to instruct the LLM to generate prompts without the LLM knowing what Stable Diffusion is; giving it the exact format to generate a prompt with has decent results. Unfortunately, the often injects plot details or non-visual content into the prompts, no matter how often you tell it not to. These details skew the relevance of the prompts and negatively impact the quality of the generated images.

### Did Work: Function Calling Capabilities

#### What Is Function Calling?
On June 13th, [OpenAI announced a huge update to the chat completions API - function calling!]( https://openai.com/blog/function-calling-and-other-api-updates) This means we can provide the chat model with a function, and the chat model will output a JSON object according to that function's parameters.

Now, the chat models can take natural language input and interpret it into a structured format that is suitable for external tools, APIs, or database queries. The chat models are designed to detect when a function needs to be called based on the user's input and can then respond with JSON that conforms to the described function's signature. 

In essence, function calling is a way to bridge the gap between unstructured language input and structured, actionable output that can be used by other systems, tools or services.

#### How FableForge Uses Functions

For our Stable Diffusion prompts, we need structured data that strictly adheres to certain rules - a function is perfect for that! Let's take a look at one of the functions we used:

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

#### LangChain and Function Calling

When we created FableForge, OpenAI had only just announced the new function calling capabilities. Since then, [LangChain](https://github.com/hwchase17/langchain) - the open-source library we use to interact with OpenAI's chat models - has added even better support for using functions. Our implementation of functions using LangChain is as follows:

1. **Define our function**: First, we define our function, like we did above with `get_visual_description_function`.
2. **Give the chat model access to our function**: Next, we call our chat model, including our function within the `functions` parameter, like so:
```python
response= self.chat([HumanMessage(content=f'{page}')],functions=get_visual_description_function)
```

3. **Parse the JSON object**: When the chat model uses our function, it provides the output as a JSON object. To convert the JSON object into a Python dictionary containing the function output, we can do the following:
```python
function_dict = json.loads(response.additional_kwargs['function_call']['arguments'])
```
In the function we defined earler, 'setting' was one of the parameters. To access this, we can write:

```python
setting = function_dict['setting']
```

And we're done! We can follow the same steps for the each of the other parameters to extract them.

### Perfecting the Process: Using Deep Lake for Storage and Analysis
The final step breakthrough for perfecting FableForge was using Deep Lake to store the generated images and text. With Deep Lake, we could store multiple modalities of data, such as image and text, in the cloud. The web-based UI provided by Deep Lake made it incredibly straightforward to display, analyze, and optimize the generated images and prompts, improving the quality of our picture book output. For future Stable Diffusion endeavours, we now have a huge dataset showing us what prompts work, and what prompts don't!

![enter image description here](https://i.imgur.com/or2MxIe.png)

## Building FableForge
FableForge consists of four main components:
1. The generation of the text and images
2. The combining of the text and images to create the book
3. Saving the images and prompts to the Deep Lake dataset
4. The UI

Let's take a look at each component individually, starting with the generation of the text and images. Here's a high-level overview of the architecture:

![enter image description here](https://user-images.githubusercontent.com/30129211/246647581-54dbaa98-5a89-4af4-8ff2-9640a40e773c.png)

## First Component: Generation

All code for this component can be found in the [`api_utils.py`](https://github.com/e-johnstonn/FableForge/blob/master/api_utils.py) file. 

1. **Text Generation**: To generate the text for the children's book, we use LangChain and the ChatOpenAI chat model. 

```python
def get_pages(self):
        pages = self.chat([HumanMessage(content=f'{self.book_text_prompt} Topic: {self.input_text}')]).content
        return pages

```
`self.book_text_prompt` is a simple prompt that instructs the model to generate a children's story. Inside the prompt, we specify the number of pages, and what format the text should come in. The full prompt can be found in the [`prompts.py`](https://github.com/e-johnstonn/FableForge/blob/master/prompts.py) file.

2. **Visual Prompts Generation**: To produce the prompts we will use with Stable Diffusion, we use functions, as outlined above. First, we send the whole book to the model:
```python
    def get_prompts(self):
        base_atmosphere = self.chat([HumanMessage(content=f'Generate a visual description of the overall lightning/atmosphere of this book using the function.'
                                                          f'{self.book_text}')], functions=get_lighting_and_atmosphere_function)
        summary = self.chat([HumanMessage(content=f'Generate a concise summary of the setting and visual details of the book')]).content

```

Since we want our book to have a consistent style throughout, we will take the contents of `base_atmosphere` and append it to each individual prompt we generate later on. To further ensure our visuals stay consistent, we generate a concise summary of the visuals of the book. This summary will be sent to the model later on, accompanying each individual page, to generate our Stable Diffusion prompts. 

```python
        def generate_prompt(page, base_dict):
            prompt = self.chat([HumanMessage(content=f'General book info: {base_dict}. Passage: {page}. Infer details about passage if they are missing, '
                                                     f'use function with inferred detailsm as if you were illustrating the passage.')],
                               functions=get_visual_description_function)
```
This method will be called for each individual page of the book. We send the model the info we just gathered along with a page from the book, and give it access to the `get_visual_description_function` function. The output of this will be a JSON object containing all the elements we need to form our prompts!

```python
    for i, prompt in enumerate(prompt_list):
        entry = f"{prompt['setting']}, {prompt['time_of_day']}, {prompt['weather']}, {prompt['key_elements']}, {prompt['specific_details']}, " \
                f"{base_dict['lighting']}, {base_dict['mood']}, {base_dict['color_palette']}, in the style of {style}"
```
Here, we combine everything together. Now that we have our prompts, we can send them to Replicate's Stable Diffusion API and get our images. Once those are downloaded, we can move onto the next step.

## Second Component: Combining

Now that we have our text and images, we could just open up MS Paint and copy paste the text onto each corresponding image. That wouldn't be the prettiest, and it's also time-consuming; instead, let's do it programmatically. In [`pdf_gen_utils.py`](https://github.com/e-johnstonn/FableForge/blob/master/pdf_gen_utils.py), we turn our ingredients into a proper book in these steps:
1. **Text Addition and Image Conversion**: First, we take each image, resize it, and apply a fading mask to the bottom - a white space for us to place our text. We then add the text to the faded area, convert it into a PDF, and save it.
2. **Cover Generation**: A book needs a cover that doesn't follow the same format as the rest of the pages. Instead of a fading mask, we take the cover image and place a white box over a portion of it for the title to be placed within. The other steps (resizing, saving as PDF) are the same as above.
3. **PDF Assembly**: Once we have all the pages completed, we combine them all into a single PDF and delete the files we no longer need.

## Third Component: Saving to Deep Lake

Now that we have our picture book all done, we want to store the images and prompts in Deep Lake. All For this, we created a `SaveToDeepLake` class:

```python
import deeplake

class SaveToDeepLake:
    def __init__(self, buildbook_instance, name=None, dataset_path=None):
        self.dataset_path = dataset_path
        try:
            self.ds = deeplake.load(dataset_path, read_only=False)
            self.loaded = True
        except:
            self.ds = deeplake.empty(dataset_path)
            self.loaded = False

        self.prompt_list = buildbook_instance.sd_prompts_list
        self.images = buildbook_instance.source_files

    def fill_dataset(self):
        if not self.loaded:
            self.ds.create_tensor('prompts', htype='text')
            self.ds.create_tensor('images', htype='image', sample_compression='png')
        for i, prompt in enumerate(self.prompt_list):
            self.ds.append({'prompts': prompt, 'images': deeplake.read(self.images[i])})


```
When initialized, the class first tries to load a Deep Lake dataset from the provided path. If the dataset doesn't exist, a new one is created. 

If the dataset already existed, we simply add the prompts and images. The images can be easily uploaded using `deeplake.read()`, as Deep Lake is built to handle multimodal data. 

If the dataset is empty, we first need to create the tensors to store our data. In this case, we create a tensor 'prompts' for our prompts, and 'images' for our images. Our images are in PNG format, so we set `sample_compression` to `'png'`. 

Once uploaded, we can view them in the UI, as shown above. 

All code can be found in the [`deep_lake_utils.py`](https://github.com/e-johnstonn/FableForge/blob/master/deep_lake_utils.py) file. 

## Final Component: Streamlit UI

To create a quick and simple UI, we used Streamlit. The complete code can be found in [`main.py`](https://github.com/e-johnstonn/FableForge/blob/master/main.py). 

Our UI has three main features:

1. **Prompt Format**: In this text input box, we allow the user to specify the prompt to generate the book based off of. This could be anything from a theme, a plot, a time period, and so on. 
2. **Book Generation**: Once the user has input their prompt, they can click the *Generate* button to generate the book. The app will run, going through all of the steps outlined above, until it completes the generation. The user will then be presented with a button to download their finished book.
3. **Saving to Deep Lake**: To save the prompts and images to their Deep Lake dataset, the user can click the *Save to Deep Lake* check-box. Once the book is generated, this will run in the background, filling the user's dataset with all their generated prompts and images. 

Streamlit is an excellent choice for quick prototyping and smaller projects such as FableForge - the entire UI is less than 60 lines of code!



## Conclusion: The Future of AI-Generated Picture Books with FableForge

Developing FableForge was a perfect example of how new AI tools and methodologies can be leveraged to overcome hurdles. Through leveraging the power of OpenAI's function calling feature, Stable Diffusion's image generation abilities, and Deep Lake's multimodal dataset storage and analysis capabilities, we were able to create an app that opens up a new frontier in children's picture book creation.

While we had our fair share of obstacles, from teaching the language model to generate image prompts to finding the perfect method of storing and analyzing data, every challenge brought valuable learning experiences and led to making FableForge into a working app!
