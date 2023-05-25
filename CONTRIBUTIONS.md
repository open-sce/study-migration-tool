# How to contribute to the Study Migration Tool

We encourage you to contribute to the Study Migration Tool!

We predominately utilise GitHub issues to track questions, feature requests and bugs. If you are unfamiliar with GitHub 
issues, please see the please see the [GitHub issues quickstart](https://docs.github.com/en/issues/tracking-your-work-with-issues/quickstart)
for more information.

**Please remember to check the Issues section before creating new ones to ensure they do not already exist!**

### **Reporting Bugs**

Open a new GitHub issue reporting the problem with an appropriate **title and clear description** of the issue and where 
you encountered it. **Please be mindful not to accidentally disclose private data in your report!**

### **Patching Bugs**

Open a new GitHub pull request with the patch and ensure the PR description clearly describes the problem and solution. 

Please also include a reference to the issue being patched if applicable.

### **Modifying existing or adding new features**

Create an issue proposing your changes and begin developing them on a new Fork. If you are unfamiliar with GitHub forks, 
please see the [GitHub fork documentation](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks).

**All new features or modifications must include appropriate new or updated unit tests.**

When your changes are ready, raise a pull request from that fork for our team to review.

### **Feature Requests**

**Please check the GitHub Issues section for issues tagged as "*enhancement*" before making a new feature request that
may have already been requested!**

Open a new GitHub issue outlining the feature with an appropriate **title and clear description**

### **File Structure**

- **assets** Contains the application theme, stylesheet (dash.css) and image files for the application
- **data** Contains the placeholder data file in csv format
- **src** Contains the data processing functionality
- *tests** Contains Pytest unit tests
- app.py is the application file containing the dash app layout and callbacks
- config.py is where any changeable variable are stored
- utils.py contains useful standalone functions