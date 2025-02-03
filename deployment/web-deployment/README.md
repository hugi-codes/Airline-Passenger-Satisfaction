## Deploy the App for Free on Streamlit Community Cloud 🌍

### Prepare Your Repository
Ensure your repository contains the following:

-`app.py`: Streamlit app code.

Before proceeding, update the model path in `app.py` to:

```
deployment/web_deployment/model.pkl
```

Streamlit requires an explicit path for recognizing the model.

### Push Your Code to GitHub
Commit and push your code to a GitHub repository.

### Deploy on Streamlit
- Go to [Streamlit Community Cloud](https://share.streamlit.io/).
- Log in with your GitHub account.
- Click on **"New app"** and select the repository containing your app.
- Configure the deployment:
  - **Main file path**: `app.py`
  - Streamlit will automatically install the dependencies from `requirements.txt`.

Happy deploying! 🚀