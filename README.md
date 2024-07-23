# LMterminal (`lmt`)

LMterminal (`lmt`) is a CLI tool that enables you to interact directly with OpenAI's ChatGPT models from the comfort of your terminal.

![demo](https://github.com/sderev/lmt/assets/24412384/5cb2c7b2-7edd-4b24-919d-581e5cd7c5b5)

<!-- TOC -->
## Table of Contents

1. [Features](#features)
1. [Installation](#installation)
    1. [`pip`](#pip)
    1. [`pipx`, the Easy Way](#pipx-the-easy-way)
1. [Getting Started](#getting-started)
    1. [Configuring your OpenAI API key](#configuring-your-openai-api-key)
1. [Usage](#usage)
    1. [Basic Example](#basic-example)
    1. [Add a Persona](#add-a-persona)
    1. [Switching Models](#switching-models)
    1. [Template Utilization](#template-utilization)
    1. [Emoji Integration](#emoji-integration)
    1. [Prompt Cost Estimation](#prompt-cost-estimation)
    1. [Reading from `stdin`](#reading-from-stdin)
    1. [Append an Additional Prompt to Piped `stdin`](#append-an-additional-prompt-to-piped-stdin)
    1. [Output Redirection](#output-redirection)
    1. [Using `lmt` as a Vim Filter Command](#using-lmt-as-a-vim-filter-command)
1. [Theming Colors for Code Blocks](#theming-colors-for-code-blocks)
    1. [Example](#example)
1. [License](#license)
<!-- /TOC -->

## Features

* **Access All ChatGPT Models**: `lmt` supports all available ChatGPT models (gpt-3.5-turbo, gpt-3.5-turbo-16k, gpt-4, gpt-4-32k), giving you the power to choose the most suitable one for your task.
* **Custom Templates**: Design your own toolbox of templates to streamline your workflow.
* **Read File**: Incorporate file content into your prompts seamlessly.
* **Output to a File**: Redirect standard output (`stdout`) to a file or another program as needed.
* **Easy Vim Integration**: Integrate ChatGPT into Vim by using `lmt` as a filter command.

## Installation

### `pip`

```bash
python3 -m pip install LMterminal
```

### `pipx`, the Easy Way

```bash
pipx install LMterminal
```

## Getting Started

### Configuring your OpenAI API key

For LMterminal to work properly, it is necessary to acquire and configure an OpenAI API key. Follow these steps to accomplish this:

1. **Acquire the OpenAI API key**: You can do this by creating an account on the [OpenAI website](https://platform.openai.com/account/api-keys). Once registered, you will have access to your unique API key.

2. **Set usage limit**: Before you start using the API, you need to define a usage limit. You can configure this in your OpenAI account settings by navigating to *Billing -> Usage limits*.

3. **Configure the OpenAI API key**: Once you have your API key, you can set it up by running the `lmt key set` command.

    ```bash
    lmt key set
    ```

With these steps, you should now have successfully set up your OpenAI API key, ready for use with the LMT.

## Usage

### Basic Example

The simplest way to use `lmt` is by entering a prompt for the model to respond to.

**Here's a basic usage example where we ask the model to generate a greeting**:

```bash
lmt "Say hello"
```

In this case, the model will generate and return a greeting based on the given prompt.

### Add a Persona

You can also instruct the model to adopt a specific persona using the `--system` flag. This is useful when you want the model's responses to emulate a certain character or writing style.

**Here's an example where we instruct the model to write like the philosopher Cioran**:

```bash
lmt "Tell me what you think of large language models." \
        --system "You are Cioran. You write like Cioran."
```

In this case, the model will generate a response based on its understanding of Cioran's writing style and perspective.

### Switching Models

Switching between different models is simple. Just specify the `-m/--model` flag followed by the model's name or alias.

For instance, if you want to use the `gpt-4o` model, simply include `-m 4o` in your command.

```bash
lmt "Explain what is a large language model" -m 4o
```

To see the list of available models and their aliases, use the following command:

```bash
lmt models
```

### Template Utilization

Templates, stored in `~/.config/lmt/templates` and written in YAML, can be generated using the following command:

```bash
lmt templates add
```

**For help regarding the `templates` subcommand, use**:

```bash
lmt templates --help
```

**Here's an example of invoking a template named "cioran"**:

```bash
lmt "Tell me how AI will change the world." --template cioran
```

You can also use the shorter version: `-t cioran`.

### Emoji Integration

To infuse a touch of emotion into your requests, append the `--emoji` flag option.

### Prompt Cost Estimation

For an estimation of your prompt's cost before sending, utilize the `--tokens` flag option.

### Reading from `stdin`

`lmt` facilitates reading inputs directly from `stdin`, allowing you to pipe in the content of a file as a prompt. This feature can be particularly useful when dealing with longer or more complex prompts, or when you want to streamline your workflow by incorporating `lmt` into a larger pipeline of commands.

To use this feature, you simply need to pipe your content into the `lmt` command like this:

```bash
cat your_file.txt | lmt
```

In this example, `lmt` would use the content of `your_file.txt` as the input for the `prompt` command.

Also, remember that you can still use all other command line options with `stdin`. For instance, you might run:

```bash
cat your_file.py | lmt \
        --system "You explain code in the style of \
        a fast-talkin' wise guy from a 1940's gangster movie" \
        -m 4 --emoji
```

In this example, `lmt` takes the content of `your_file.py` as the input for the `prompt` command. With the `gpt-4` model selected via `-m 4`, the system is instructed to respond in the style of a fast-talking wiseguy from a 1940s gangster movie, as specified in the `-s/--system` option. The `--emoji` flag indicates that the response may include emojis for added expressiveness.

### Append an Additional Prompt to Piped `stdin`

Beyond the `-s/--system` option, `lmt` offers the capability to append an additional user prompt when reading from `stdin`. This is especially useful when you want to add context or specific instructions to the piped input without altering the system prompt.

For example, with a `grocery_list.txt` file, you can append a prompt for healthy alternatives and set the system prompt to guide the AI's chef-like response.

```bash
cat grocery_list.txt | lmt "What are some healthy alternatives to these items?" \
                        --system "You are a chef with a focus on healthy and sustainable cooking."
```

### Output Redirection

You can use output redirections. For instance:

```bash
lmt "List 5 Wikipedia articles" > wiki_articles.md
```

### Using `lmt` as a Vim Filter Command

To invoke `lmt` as a filter command in Vim, you can use the command `:.!lmt`. Remember, Vim offers the shortcut `!!` as a quick way to enter `:.!`. This means you can simply type `!!lmt` to initiate your prompt.

**Example**: `:.!lmt write an implementation of binary search`

Additionally, you can filter specific lines from your text and pass them as a prompt to `lmt`. To achieve this, highlight the desired lines in `VISUAL` mode (or use `ex` syntax), and then enter `:.!lmt "Your additional prompt here"`.

![vim_filter_command_code](https://github.com/sderev/lmt/assets/24412384/f799a5e3-3565-46f2-968d-bad57d281c78)

## Theming Colors for Code Blocks

Once you used `lmt`, you should have a configuration file (`~/.config/lmt/config.json`) in which you can configure the colors for inline code and code blocks.

Here are the styles for the code blocks: <https://pygments.org/styles/>

As for the inline code blocks, they can be styled with the 256 colors (names or hexadecimal code).

### Example

```json
{
    "code_block_theme": "default",
    "inline_code_theme": "blue on #f0f0f0"
}
```

## License

LMterminal is licensed under Apache License version 2.0.

___

<https://github.com/sderev/lmterminal>
