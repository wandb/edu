const { Octokit } = require("@octokit/rest");
// install with npm install @octokit/rest

const octokit = new Octokit({
auth: process.env.GITHUB_TOKEN
});

// Create an issue commment
octokit.issues.createComment({
    issue_number: 6,
    owner: 'hamelsmu',
    repo: 'wandb-cicd',
    body: "Hello! I'm making a comment from `octokit.js!`"
});

