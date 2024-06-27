# login

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_login.py#L46-L103' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


Set up W&B login credentials.

```python
login(
    anonymous: Optional[Literal['must', 'allow', 'never']] = None,
    key: Optional[str] = None,
    relogin: Optional[bool] = None,
    host: Optional[str] = None,
    force: Optional[bool] = None,
    timeout: Optional[int] = None,
    verify: bool = (False)
) -> bool
```

By default, this will only store the credentials locally without
verifying them with the W&B server. To verify credentials, pass
verify=True.

| Arguments |  |
| :--- | :--- |
|  `anonymous` |  (string, optional) Can be "must", "allow", or "never". If set to "must" we'll always log in anonymously, if set to "allow" we'll only create an anonymous user if the user isn't already logged in. |
|  `key` |  (string, optional) authentication key. |
|  `relogin` |  (bool, optional) If true, will re-prompt for API key. |
|  `host` |  (string, optional) The host to connect to. |
|  `force` |  (bool, optional) If true, will force a relogin. |
|  `timeout` |  (int, optional) Number of seconds to wait for user input. |
|  `verify` |  (bool) Verify the credentials with the W&B server. |

| Returns |  |
| :--- | :--- |
|  `bool` |  if key is configured |

| Raises |  |
| :--- | :--- |
|  AuthenticationError - if api_key fails verification with the server UsageError - if api_key cannot be configured and no tty |
