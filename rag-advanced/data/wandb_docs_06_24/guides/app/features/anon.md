---
description: Log and visualize data without a W&B account
displayed_sidebar: default
---

# Anonymous Mode

Are you publishing code that you want anyone to be able to run easily? Use Anonymous Mode to let someone run your code, see a W&B dashboard, and visualize results without needing to create a W&B account first.

Allow results to be logged in Anonymous Mode with `wandb.init(`**`anonymous="allow"`**`)`

:::info
**Publishing a paper?** Please [cite W&B](https://docs.wandb.ai/company/academics#bibtex-citation), and if you have questions about how to make your code accessible while using W&B, reach out to us at support@wandb.com.
:::

### How does someone without an account see results?

If someone runs your script and you have to set `anonymous="allow"`:

1. **Auto-create temporary account:** W&B checks for an account that's already signed in. If there's no account, we automatically create a new anonymous account and save that API key for the session.
2. **Log results quickly:** The user can run and re-run the script, and automatically see results show up in the W&B dashboard UI. These unclaimed anonymous runs will be available for 7 days.
3. **Claim data when it's useful**: Once the user finds valuable results in W&B, they can easily click a button in the banner at the top of the page to save their run data to a real account. If they don't claim a run, it will be deleted after 7 days.

:::caution
**Anonymous run links are sensitive**. These links allow anyone to view and claim the results of an experiment for 7 days, so make sure to only share links with people you trust. If you're trying to share results publicly, but hide the author's identity, please contact us at support@wandb.com to share more about your use case.
:::

### What happens to users with existing accounts?

If you set `anonymous="allow"` in your script, we will check to make sure there's not an existing account first, before creating an anonymous account. This means that if a W&B user finds your script and runs it, their results will be logged correctly to their account, just like a normal run.

### What are features that aren't available to anonymous users?

*   **No persistent data**: Runs are only saved for 7 days in an anonymous account. Users can claim anonymous run data by saving it to a real account.


![](@site/static/images/app_ui/anon_mode_no_data.png)

*   **No artifact logging**: Runs will print a warning on the command line that you can't log an artifact to an anonymous run.

![](@site/static/images/app_ui/anon_example_warning.png)

* **No profile or settings pages**: Certain pages aren't available in the UI, because they're only useful for real accounts.

## Example usage

[Try the example notebook](http://bit.ly/anon-mode) to see how anonymous mode works.

```python
import wandb

# Start a run allowing anonymous accounts
wandb.init(anonymous="allow")

# Log results from your training loop
wandb.log({"acc": 0.91})

# Mark the run as finished
wandb.finish()
```
