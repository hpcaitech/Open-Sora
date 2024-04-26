# Contributing

The Open-Sora project welcomes any constructive contribution from the community and the team is more than willing to work on problems you have encountered to make it a better project.

## Development Environment Setup

To contribute to Open-Sora, we would like to first guide you to set up a proper development environment so that you can better implement your code. You can install this library from source with the `editable` flag (`-e`, for development mode) so that your change to the source code will be reflected in runtime without re-installation.

You can refer to the [Installation Section](./README.md#installation) and replace `pip install -v .` with `pip install -v -e .`.

### Code Style

We have some static checks when you commit your code change, please make sure you can pass all the tests and make sure the coding style meets our requirements. We use pre-commit hook to make sure the code is aligned with the writing standard. To set up the code style checking, you need to follow the steps below.

```shell
# these commands are executed under the Open-Sora directory
pip install pre-commit
pre-commit install
```

Code format checking will be automatically executed when you commit your changes.

## Contribution Guide

You need to follow these steps below to make contribution to the main repository via pull request. You can learn about the details of pull request [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).

### 1. Fork the Official Repository

Firstly, you need to visit the [Open-Sora repository](https://github.com/hpcaitech/Open-Sora) and fork into your own account. The `fork` button is at the right top corner of the web page alongside with buttons such as `watch` and `star`.

Now, you can clone your own forked repository into your local environment.

```shell
git clone https://github.com/<YOUR-USERNAME>/Open-Sora.git
```

### 2. Configure Git

You need to set the official repository as your upstream so that you can synchronize with the latest update in the official repository. You can learn about upstream [here](https://www.atlassian.com/git/tutorials/git-forks-and-upstreams).

Then add the original repository as upstream

```shell
cd Open-Sora
git remote add upstream https://github.com/hpcaitech/Open-Sora.git
```

you can use the following command to verify that the remote is set. You should see both `origin` and `upstream` in the output.

```shell
git remote -v
```

### 3. Synchronize with Official Repository

Before you make changes to the codebase, it is always good to fetch the latest updates in the official repository. In order to do so, you can use the commands below.

```shell
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

### 5. Create a New Branch

You should not make changes to the `main` branch of your forked repository as this might make upstream synchronization difficult. You can create a new branch with the appropriate name. General branch name format should start with `hotfix/` and `feature/`. `hotfix` is for bug fix and `feature` is for addition of a new feature.

```shell
git checkout -b <NEW-BRANCH-NAME>
```

### 6. Implementation and Code Commit

Now you can implement your code change in the source code. Remember that you installed the system in development, thus you do not need to uninstall and install to make the code take effect. The code change will be reflected in every new PyThon execution.
You can commit and push the changes to your local repository. The changes should be kept logical, modular and atomic.

```shell
git add -A
git commit -m "<COMMIT-MESSAGE>"
git push -u origin <NEW-BRANCH-NAME>
```

### 7. Open a Pull Request

You can now create a pull request on the GitHub webpage of your repository. The source branch is `<NEW-BRANCH-NAME>` of your repository and the target branch should be `main` of `hpcaitech/Open-Sora`. After creating this pull request, you should be able to see it [here](https://github.com/hpcaitech/Open-Sora/pulls).

The Open-Sora team will review your code change and merge your code if applicable.

## FQA

1. `pylint` cannot recognize some members:

Add this into your `settings.json` in VSCode:

```json
"pylint.args": [
 "--generated-members=numpy.* ,torch.*,cv2.*",
],
```
