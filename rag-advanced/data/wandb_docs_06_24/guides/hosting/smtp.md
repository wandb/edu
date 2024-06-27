---
displayed_sidebar: default
---

# SMTP Configuration

In W&B server, adding users to the instance or team will trigger an email invite. To send these email invites, W&B uses a third-party mail server. In some cases, organizations might have strict policies on traffic leaving the corporate network and hence causing these email invites to never be sent to the end user. W&B server offers an option to configure sending these invite emails via an internal SMTP server.

To configure, follow the steps below:

- Set the `GORILLA_EMAIL_SINK` environment variable in the docker container or the kubernetes deployment to `smtp://<user:password>@smtp.host.com:<port>`
- `username` and `password` are optional
- If you’re using an SMTP server that’s designed to be unauthenticated you would just set the value for the environment variable like `GORILLA_EMAIL_SINK=smtp://smtp.host.com:<port>`
- Commonly used port numbers for SMTP are ports 587, 465 and 25. Note that this might differ based on the type of the mail server you're using.
