# Fake News Detector · Docker Compose aur Jenkins practical

Tumhara Streamlit app ab har analysis ka result PostgreSQL mein save karta hai. Is practical mein pehle app apne Windows laptop par kholo. Phir GitHub par ek change push karke dikhao ki Jenkins khud tests chalata hai aur updated app usi laptop par deploy karta hai.

Docker Desktop, Git aur Jenkins pehle se ready hon to **35–55 minute** ka target rakho. Pehli baar Docker images aur Python packages download karne ka time internet speed par depend karega. Saare terminal commands **Windows PowerShell** ke liye hain.

## 1. Project folder aur Docker kholo

1. ZIP par right-click → **Extract All**. Andar woh folder kholo jahan `compose.yaml`, `Dockerfile`, `app.py` aur `Jenkinsfile` saath dikhte hain.
2. File Explorer ke address bar mein `powershell` type karke **Enter** dabao. Terminal isi folder mein khulega.
3. Docker Desktop start karo. PowerShell mein yeh commands ek-ek karke chalao.

```powershell
docker version
docker compose version
git --version
```

`docker version` mein **Client** aur **Server** dono dikhne chahiye. Jenkins dashboard bhi kholo. [http://localhost:8080](http://localhost:8080) Docker install karna ho to apne Windows build ke steps yahan milenge. [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/)

## 2. Dono containers ek saath chalao

Pehla block project ki private `.env` file mein database password banati hai. Ye command dobara chalane par pehle se bani file ko rakh leti hai.

```powershell
if (!(Test-Path .env)) {
    $dbPassword = [guid]::NewGuid().ToString('N')
    "POSTGRES_PASSWORD=$dbPassword" | Set-Content -Encoding ascii .env
}
docker compose up -d --build --wait
docker compose ps
```

`web` aur `db` ki rows mein **healthy** dikhega. `web` app chala raha hai aur `db` PostgreSQL. Browser mein app kholo. [http://localhost:8501](http://localhost:8501)

## 3. Ek prediction save karke database mein dekho

Yeh sample app mein paste karo aur **Analyze Article** dabao.

> The city council approved the budget on Wednesday after a public meeting. The plan funds a new library and expands bus service next year.

Is ZIP ke model ke saath is sample ka expected label `REAL` hai. **Prediction history · latest 10** mein uski row milegi. Neeche wali command wahi data seedha PostgreSQL se padhti hai.

```powershell
$sql = "SELECT id, label, article_preview FROM predictions ORDER BY id DESC LIMIT 3"
docker compose exec -T db psql -U newsapp -d news -c $sql
```

**History check karne ke liye** neeche ki commands chalao, phir app refresh karo. Pehle wali row ab bhi dikhni chahiye. PostgreSQL ka data `pgdata` volume mein hai. `down` ke saath `-v` lagane se volume delete hota hai.

```powershell
docker compose down
docker compose up -d --wait --pull never
```

## 4. Project GitHub par push karo

**Naya practical repo banane ke liye** GitHub → **+ → New repository**. Naam `fake-news-detector-devops` rakh sakte ho. Visibility apni preference se chuno. README, `.gitignore` aur license wale options khaali rakho. **Create repository** ke baad HTTPS URL copy karo. Apne project wale PowerShell mein commands ek-ek karke chalao. URL maange to copied URL paste karke Enter dabao.

```powershell
git init -b main
git add .
git status --short
git commit -m "Add Compose database and Jenkins deployment"
$repoUrl = Read-Host "Paste your GitHub repository HTTPS URL"
git remote add origin $repoUrl
git push -u origin main
```

`git status` mein `.env.example` dikh sakti hai. Private `.env` ko `.gitignore` exclude karta hai. Sign-in prompt aaye to apna GitHub account chuno. GitHub refresh karo. Repo ki top level par `Jenkinsfile`, `compose.yaml`, `Dockerfile`, `tests/` aur dono `.pkl` files dikhni chahiye.

**Agar project ka GitHub repo pehle se hai**, clone kiya hua folder kholo. Clone ki root mein updated ZIP ke project files copy karo, hidden files bhi. `.env` Step 2 wale folder mein hi rahegi. Jenkins ke liye password copy karte waqt wahi PowerShell use karo. `git status` mein changes dekho, phir `git add .`, `git commit -m "Add Compose and Jenkins"` aur `git push` chalao. Neeche Jenkins mein isi repo ki branch select karna. [GitHub ka Windows guide](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github?platform=windows)

## 5. Jenkins ko Docker aur database se connect karo

**Windows node ko label do.** Jenkins → **Manage Jenkins → Nodes** → apna local Windows node → **Configure**. Agar pehle ke practical built-in node par chalte the, wahi node kholo. **Labels** mein existing labels ke baad space dekar `docker-lab` add karo → **Save**. Yeh naam Jenkins ko batata hai ki build kis laptop par chalani hai.

**Database password save karo.** Project wale PowerShell mein yeh line chalao. Step 2 ka password clipboard mein aa jayega.

```powershell
(ConvertFrom-StringData (Get-Content .env -Raw)).POSTGRES_PASSWORD | Set-Clipboard
```

Jenkins → **Manage Jenkins → Credentials → System → Global credentials (unrestricted) → Add Credentials**. Kind **Secret text**, Scope **Global**, Secret mein **Ctrl+V**, aur ID mein exactly `fake-news-db-password` daalo. **Create** dabao. Jenkins aur manual Compose ab same database password use karenge.

**Private GitHub repo ke liye** Jenkins ko repo padhne ka access bhi chahiye. Existing GitHub read credential ho to Step 6 mein select karo. Naya banana ho to GitHub → **Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token**. Selected repository chuno aur **Contents → Read-only** do. Jenkins ke **Add Credentials** mein Kind **Username with password**, Username apna GitHub username, Password mein token aur ID `github-read` rakho. Public repo ka checkout anonymous chal sakta hai. [GitHub tokens](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) · [Jenkins Git credentials](https://plugins.jenkins.io/git/#plugin-content-using-credentials)

## 6. GitHub se Jenkins Pipeline chalao

Jenkins dashboard → **New Item** → naam `Fake-News-DevOps` → **Pipeline** → **OK**. Neeche **Pipeline** section mein ye fields bharo.

| Field | Value |
| --- | --- |
| Definition | `Pipeline script from SCM` |
| SCM | `Git` |
| Repository URL | Step 4 wala HTTPS URL |
| Credentials | Private repo ke liye GitHub credential, public repo ke liye `None` |
| Branch Specifier | `*/main` ya tumhari actual branch, jaise `*/master` |
| Script Path | `Jenkinsfile` |

**Save → Build Now → build number → Console Output** kholo. Docker Desktop aur Jenkins build ke dauran chalte rehne chahiye.

| Stage | Kya hota hai | Kis cheez se verify karoge |
| --- | --- | --- |
| Checkout | Jenkins GitHub se commit leta hai | Console mein checkout ka commit |
| Build image | Docker image banata hai. Image ke andar pip packages install karta hai | `fake-news-detector-web` image par commit aur build number ka tag |
| Test model and database | Teen Python tests temporary PostgreSQL se chalte hain | `Ran 3 tests` aur `OK` |
| Deploy and verify | Jenkins wahi tested image persistent `web` aur `db` ke saath chalata hai | `Web to PostgreSQL OK`, dono services healthy aur `Finished: SUCCESS` |

Test fail ho to Jenkins deploy stage skip karta hai. Jenkins isi laptop ke Docker Desktop mein image build aur deploy karta hai. [Jenkins Pipeline setup](https://www.jenkins.io/doc/book/pipeline/getting-started/#defining-a-pipeline-in-scm)

## 7. Automatic deployment ka proof dikhao

Project folder mein `notepad app.py` kholo. `Article Input</div>` search karo aur sirf displayed text `Article Input` ko `News Article Input` karo. File save karo. Phir chalao.

```powershell
git add app.py
git commit -m "Update the article input heading"
git push
```

Pehli manual build ke baad Jenkins har 2 minute GitHub par changes check karta hai. Agli build khud start hogi. Success aate hi browser mein app refresh karo. Nayi heading aur purani prediction history ek saath dikhengi. Agli build queue mein ho to agent available hote hi chalegi. [Jenkins SCM polling](https://www.jenkins.io/doc/book/pipeline/syntax/#triggers)

**Practical ke liye ye screenshots lo** — (1) `docker compose ps` mein dono healthy services, (2) app result aur prediction history, (3) SQL output, (4) GitHub repo ki files, (5) automatically triggered Jenkins build ka `Finished: SUCCESS` aur (6) browser mein nayi heading.

## Agar koi step atke

| Kya dikh raha hai | Kya karo |
| --- | --- |
| `docker` command nahi mil rahi ya `Server` nahi dikh raha | Docker Desktop kholo, Linux containers mode aur engine ready hone tak wait karo. PowerShell dobara kholo aur `docker version` chalao. |
| `POSTGRES_PASSWORD` missing ya password authentication failed | Manual commands `.env` wale folder se chalao. Jenkins credential mein usi file ka original password paste karo. PostgreSQL pehli initialization ka password use karta hai. |
| `port 8501 is already allocated` | Purani local Streamlit window mein `Ctrl+C` dabao. `docker ps` se dekho ki kaunsa container port use kar raha hai. Agar dono apps chalani hain to `.env` mein `WEB_PORT=8502` aur Jenkins node ke **Node Properties → Environment variables** mein `WEB_PORT` ki value `8502` set karo. Phir port `8502` kholo. |
| `Still waiting to schedule task` | **Manage Jenkins → Nodes** mein dekho ki `docker-lab` label wala Windows node online hai aur uska ek executor available hai. |
| PowerShell mein Docker chalta hai, Jenkins mein `access denied` ya Docker pipe nahi milti | Neeche same Windows login wala agent banao. |
| `Jenkinsfile` nahi mil raha | Repo ki root mein `Jenkinsfile` ki spelling, uppercase J, branch aur Jenkins ka Script Path check karo. |
| Ek container unhealthy hai | Project folder mein `docker compose logs --tail 40 web db` chalao. Jenkins fail ho to **Console Output** mein pehle failed stage ki error dekho. |

**Docker access ki error theek karne ke liye** usi Windows login se Jenkins agent connect karo jismein Docker Desktop chal raha hai.

1. Jenkins → **Manage Jenkins → Nodes → New Node**. Naam `windows-docker`, type **Permanent Agent**.
2. **Remote root directory** mein apne user ke writable folder ka full path do, jaise `C:\Users\Shaurya\jenkins-agent`. **Labels** mein `docker-lab`, **Number of executors** mein `1` rakho.
3. **Launch method → Launch agent by connecting it to the controller** chuno. **Use WebSocket** option dikhe to select karo. Save karo.
4. Node ke page par Windows ke liye dikhne wali download aur `java -jar agent.jar ...` commands ko Docker chalane wale login ki PowerShell mein chalao. `java -version` available hona chahiye. Ye terminal khula rakho. Agent command mein private secret hota hai, use screenshot se bahar rakho.
5. Naya node **online** dikhne par purane node ke labels se sirf `docker-lab` hatao. Job par **Build Now** dabao.

Demo ke baad containers temporarily band karne ke liye project folder se `docker compose stop` chalao. Agli baar `docker compose start` se wahi containers aur history khul jayenge.
