# Private Streamlit Cloud Test: Atom Detection

This guide is for `apps/atom_detection_ui.py`. Community Cloud runs from the repository root and reads `apps/requirements.txt` next to this entrypoint before a root dependency file. The requirements install the local `autotem-insight` package with its `ui` extra, plus the app's direct Altair and pandas imports. Streamlit and the drawable canvas are pinned to the versions used in local testing.

## Before deployment

- Use a GitHub **private repository** containing the branch with the app, every local module it imports, and its deployment requirements. A private app does not make source code in a public repository private.
- The person creating the app must have **admin** permission on that repository. Member C's collaborator/write access alone is insufficient; ask the repository owner/admin to deploy or arrange an approved private copy. Connect that GitHub account to [Streamlit Community Cloud](https://share.streamlit.io/) and authorize private-repository access (and organization access, if applicable).
- Community Cloud currently allows one private app at a time. Use test data cleared for cloud upload; do not put microscopy datasets or credentials in the repository.
- Each tester needs only an email address for the invitation. Do not collect their passwords. Cloud uploads are processed on the server; start with small, non-sensitive images, and use a separately controlled server for data that cannot leave the lab.

## Local check

From the repository root, with Python 3.12 selected:

```bash
python -m pip install -r apps/requirements.txt
python -m streamlit run apps/atom_detection_ui.py
```

Open the local app and try a permitted sample image, atom detection, and a CSV download.

## Deploy and share (repository admin)

1. Sign in at [share.streamlit.io](https://share.streamlit.io/), select the workspace matching the repository owner, and choose **Create app**.
2. Select the private repository and branch, then set the entrypoint to `apps/atom_detection_ui.py` (forward slashes). In **Advanced settings**, select Python 3.12. Deploy.
3. In **App settings > Sharing**, verify **Only specific people can view this app**. A private repository makes the app private by default. Keep this setting private for the test.
4. Use the app's **Share** control to invite approved testers by email. Invitees sign in with Google or a single-use emailed link. Invited viewers can also see app analytics and invite other viewers, so limit invitations to trusted testers.
5. Confirm an invited tester can open the app and run a small, non-sensitive sample. If the build or run fails, inspect **Manage app** logs with a repository writer/admin account.

No deployment or invitation is performed by adding these files. See the official Streamlit guides for [file organization](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization), [dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies), [GitHub permissions](https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/connect-your-github-account), [deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), and [private sharing](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app).
