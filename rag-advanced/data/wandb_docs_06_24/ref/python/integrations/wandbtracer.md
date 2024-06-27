# WandbTracer



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L99-L281)



Callback Handler that logs to Weights and Biases.

```python
WandbTracer() -> Any
```




This handler will log the model architecture and run traces to Weights and Biases. This will ensure that all LangChain activity is logged to W&B.



| Attributes | |
| :--- | :--- |
| `always_verbose` | Whether to call verbose callbacks even if verbose is False. |
| `ignore_agent` | Whether to ignore agent callbacks. |
| `ignore_chain` | Whether to ignore chain callbacks. |
| `ignore_llm` | Whether to ignore LLM callbacks. |



## Methods

### `finish`



[View source](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L152-L162)

```python
@staticmethod
finish() -> None
```
Waits for all asynchronous processes to finish and data to upload.

<!-- Stops watching all LangChain activity and resets the default handler.

It is recommended to call this function before terminating the kernel or
python script. -->

### `finish_run`



[View source](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L202-L211)

```python
finish_run() -> None
```

Waits for W&B data to upload.


### `init`



[View source](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L111-L150)

```python
@classmethod
init(
 run_args: Optional[WandbRunArgs] = None,
 include_stdout: bool = (True),
 additional_handlers: Optional[List['BaseCallbackHandler']] = None
) -> None
```

Sets up a WandbTracer and makes it the default handler.


#### Parameters:


* **`run_args`**: (dict, optional) Arguments to pass to `wandb.init()`. If not provided, `wandb.init()` will be
 called with no arguments. Please refer to the `wandb.init` for more details.
* **`include_stdout`**: (bool, optional) If True, the `StdOutCallbackHandler` will be added to the list of
 handlers. This is common practice when using LangChain as it prints useful information to stdout.
* **`additional_handlers`**: (list, optional) A list of additional handlers to add to the list of LangChain handlers.

To use W&B to
monitor all LangChain activity, simply call this function at the top of
the notebook or script:
```
from wandb.integration.langchain import WandbTracer
WandbTracer.init()
# ...
# end of notebook / script:
WandbTracer.finish()
```.

It is safe to call this repeatedly with the same arguments (such as in a
notebook), as it will only create a new run if the run_args differ.

### `init_run`



[View source](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L164-L200)

```python
init_run(
 run_args: Optional[WandbRunArgs] = None
) -> None
```

Initialize wandb if it has not been initialized.


#### Parameters:


* **`run_args`**: (dict, optional) Arguments to pass to `wandb.init()`. If not provided, `wandb.init()` will be
 called with no arguments. Please refer to the `wandb.init` for more details.

We only want to start a new run if the run args differ. This will reduce
the number of W&B runs created, which is more ideal in a notebook
setting. Note: it is uncommon to call this method directly. Instead, you
should use the `WandbTracer.init()` method. This method is exposed if you
want to manually initialize the tracer and add it to the list of handlers.

### `load_default_session`



[View source](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L264-L267)

```python
load_default_session() -> "TracerSession"
```

Load the default tracing session and set it as the Tracer's session.


### `load_session`



[View source](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/langchain/wandb_tracer.py#L259-L262)

```python
load_session(
 session_name: str
) -> "TracerSession"
```

Load a session from the tracer.


### `new_session`



```python
new_session(
 name: Optional[str] = None,
 **kwargs
) -> TracerSession
```

NOT thread safe, do not call this method from multiple threads.


### `on_agent_action`



```python
on_agent_action(
 action: AgentAction,
 **kwargs
) -> Any
```

Do nothing.


### `on_agent_finish`



```python
on_agent_finish(
 finish: AgentFinish,
 **kwargs
) -> None
```

Handle an agent finish message.


### `on_chain_end`



```python
on_chain_end(
 outputs: Dict[str, Any],
 **kwargs
) -> None
```

End a trace for a chain run.


### `on_chain_error`



```python
on_chain_error(
 error: Union[Exception, KeyboardInterrupt],
 **kwargs
) -> None
```

Handle an error for a chain run.


### `on_chain_start`



```python
on_chain_start(
 serialized: Dict[str, Any],
 inputs: Dict[str, Any],
 **kwargs
) -> None
```

Start a trace for a chain run.


### `on_llm_end`



```python
on_llm_end(
 response: LLMResult,
 **kwargs
) -> None
```

End a trace for an LLM run.


### `on_llm_error`



```python
on_llm_error(
 error: Union[Exception, KeyboardInterrupt],
 **kwargs
) -> None
```

Handle an error for an LLM run.


### `on_llm_new_token`



```python
on_llm_new_token(
 token: str,
 **kwargs
) -> None
```

Handle a new token for an LLM run.


### `on_llm_start`



```python
on_llm_start(
 serialized: Dict[str, Any],
 prompts: List[str],
 **kwargs
) -> None
```

Start a trace for an LLM run.


### `on_text`



```python
on_text(
 text: str,
 **kwargs
) -> None
```

Handle a text message.


### `on_tool_end`



```python
on_tool_end(
 output: str,
 **kwargs
) -> None
```

End a trace for a tool run.


### `on_tool_error`



```python
on_tool_error(
 error: Union[Exception, KeyboardInterrupt],
 **kwargs
) -> None
```

Handle an error for a tool run.


### `on_tool_start`



```python
on_tool_start(
 serialized: Dict[str, Any],
 input_str: str,
 **kwargs
) -> None
```

Start a trace for a tool run.




