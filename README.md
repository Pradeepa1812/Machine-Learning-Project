##  END to END Machine Learning Project
<pre>
1)Setting up github Repository

Upgrade pip - pip install --upgrade pip
Create New Environment - 
      conda create -p venv python==3.8 -y 
      conda activate venv/
      
Sync github with vscode
      git init
      create readme.md file
      git add README.md
      git commit -m "first commit"
      git branch -M main
      git remote add origin https://github.com/Pradeepa1812/Machine-Learning-Project.git
      git push -u origin main
      git pull

.gitignore created via github create new file with python template 

creating setup.py and requirements.txt file
setup.py - Responsible for creating ML application as a package
           we can install this package (even deploy in pypi) in our projects and can able to use
           Building our application as a package itself

creating  src/ __init__.py