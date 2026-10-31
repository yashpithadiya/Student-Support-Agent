# Intelligent Student Support Agent

> **Demonstration notice:** All student identities, records, fee amounts, scholarships, tickets and handbook policies in this repository are fictional sample data. They are not official university records or regulations.

## Project overview

The Intelligent Student Support Agent is a Flask-based application that helps students with common academic and administrative questions. Instead of following one fixed sequence, the agent decides for each message whether to answer directly, use a student-data tool, retrieve a relevant policy section, use an allowed memory item, prepare a support ticket, or ask the student for missing information.

Task 2 extends the application with authenticated student and administrator roles, server-side authorization, student-data privacy controls, approval-based record changes, and an audit log.

## Main capabilities

### TASK 1 — Agent behaviour and observability

- Framework-free ReAct-style agent loop using the Groq SDK and `openai/gpt-oss-20b` by default.
- Tool calling for fee status, scholarship status, student records, support-ticket preparation, policy search and memory.
- Policy retrieval from heading-based handbook sections using keyword/IDF scoring.
- Persistent, relevance-filtered memory with a code-level sensitive-data filter.
- Context selection using relevant memory and recent conversation turns.
- Structured execution traces with tool calls, retrieved evidence, memory references, latency, token usage and errors.
- Golden-dataset evaluation and automated tests.

### TASK 2 — Identity, privacy and human approval

- Separate **Student Login** and **Admin Login** experiences.
- Password hashes stored in SQLite; role-aware server-side route protection.
- Student identity is taken from the authenticated server session, not trusted from chat text.
- Requests for another student's private information are refused before model/data access; the tool layer also enforces student ownership.
- Changes to a student's own record are submitted as pending requests and require administrator approval.
- Support-ticket creation uses a draft-and-confirm flow; the agent prepares the draft and the student confirms submission.
- Administrator dashboard for overview, approval requests, student records, support tickets, policies, agent traces and audit log.
- Audit entries for important authentication, access-control, ticket and approval actions.

## Architecture

The diagram below shows the main runtime components and the separation between the student and administrator interfaces.

![Student Support Agent architecture](architecture.png)

## Technology stack

| Layer | Technology |
|---|---|
| Application / UI | Python, Flask, HTML, CSS, JavaScript |
| Agent orchestration | Custom Python ReAct-style loop |
| Language model | Groq API, `openai/gpt-oss-20b` by default |
| Student and policy data | CSV and Markdown sample data |
| Persistent storage | SQLite |
| Memory | SQLite with an allow-list and sensitive-data checks |
| Retrieval | Heading-based chunks and keyword/IDF scoring |
| Tracing | Structured JSONL traces |
| Evaluation | Python evaluation runner, CSV results and Markdown summary |
| Demo credential workbook | openpyxl (used when demo accounts are seeded) |
| Testing | pytest |

## Repository structure

```text
.
├── app.py                         # Flask application and routes
├── agent.py                       # Agent loop, context selection, privacy pre-check
├── llm.py                          # Groq model wrapper and error handling
├── tools.py                        # Tool definitions, execution and tool-level checks
├── memory.py                       # Persistent memory and sensitive-data filtering
├── retrieval.py                    # Student-handbook chunking and retrieval
├── tracing.py                      # Per-run trace and measurements
├── database.py                     # SQLite persistence, approvals, tickets and audit
├── accounts.py                     # Demo account seeding and password verification
├── config.py                       # Environment and runtime configuration
├── templates/
│   ├── login.html                   # Student/Admin login tabs
│   ├── index.html                   # Student application interface
│   └── admin.html                   # Administrator dashboard
├── data/
│   ├── sample_students.csv          # Fictional student records
│   ├── student_policy.md            # Fictional sample handbook
│   └── golden_dataset.json         # Evaluation cases
├── evaluation/
│   ├── evaluate.py                  # Evaluation runner
│   └── mock_llm.py                  # Mock model for harness testing
├── tests/                           # Automated tests
├── docs/                            # Submission and recording guidance
├── .env.example                     # Environment-variable template
└── requirements.txt
```

Runtime files such as `data/agent.db` and `traces/` are created locally and should not be committed.

## Setup

### 1. Requirements

- Python 3.10 or later (use the Python version supported by your installed dependencies).
- A Groq API key for live model responses.
- Git for version control.

### 2. Create and activate a virtual environment

**Windows PowerShell**
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install openpyxl  # used to generate the local demo-accounts workbook
```

### 4. Configure the API key

Copy `.env.example` to `.env`, then add your own key:

```env
GROQ_API_KEY=your_groq_api_key_here
# Optional:
# MODEL=openai/gpt-oss-20b
# MAX_TURNS=6
# HISTORY_TURNS=3
```

Never share `.env`, paste the key into a screenshot, or commit it to GitHub. The `.env` file is excluded by `.gitignore`.

## Run the application

From the project root:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

The login screen has separate tabs for Student Login and Admin Login. Use the locally generated demo credentials from `testing/Demo_Test_Accounts.xlsx`. Keep that workbook private and do not upload it to a public repository.

On a first run, the application seeds demo accounts from the fictional student data and writes the generated credentials to the local testing workbook.

## Interface guide

### Student interface

| Screen / section | Purpose |
|---|---|
| New Chat | Ask questions and interact with the agent |
| Run Details | Inspect tool calls, retrieved evidence, memory used, model/token metrics and the full trace for a response |
| History | Reopen or delete the student's own conversations |
| Student Info | View the signed-in student's sample profile and academic/fee information |
| Support | View own tickets and requests; submit or revoke eligible items |
| Policies | Browse the sample handbook sections and download the source policy file |
| Traces | Inspect the student's agent execution traces and timeline |
| Profile / Settings | View account details, change appearance and manage remembered facts |

### Administrator interface

| Screen / section | Purpose |
|---|---|
| Overview | Summary of pending approvals, open tickets, student count and agent activity |
| Approval Requests | Review pending student record-change requests; approve or reject with a reason when rejecting |
| Student Records | Search and inspect sample student records |
| Support Tickets | Review tickets and update their status |
| Policies | Browse the sample handbook |
| Agent Traces | Inspect persisted agent runs |
| Audit Log | Review important actions and their outcomes |

The evaluation dashboard is not a student-facing screen. Evaluation is run from the command line; administrator evaluation API endpoints remain available in the backend.

## How the agent works

At a high level, each request follows this loop:

1. The application receives the student's message.
2. The agent selects relevant context: system instructions, allowed memory relevant to the question, recent conversation turns and the current question.
3. The model decides whether to answer or request one or more tools.
4. Python validates and executes requested tools.
5. Tool observations are returned to the model for the next decision.
6. The loop ends with a final answer, a turn limit, or a handled model error.
7. A trace records the run and its measurements.

The model requests tools; application code executes them. The sequence is not hard-coded as one fixed workflow.

## Privacy and approval controls

- The server session determines the authenticated user and role.
- Student routes and administrator routes are protected by server-side role checks.
- The agent checks for requests about another student before making a model call.
- Tool execution independently checks that student-data access belongs to the signed-in student.
- Sensitive memory values such as passwords, OTPs, tokens and other secret-like values are rejected by code.
- A record-change request remains pending until an administrator approves it.
- Approval status, record update and audit entry are handled together in a database transaction.
- Support tickets are prepared as drafts and are created only after student confirmation.
- Rejected or revoked requests do not apply the requested record change.

## Evaluation and tests

Run the automated tests:

```bash
python -m pytest -q
```

Run the live evaluation using your configured Groq API key:

```bash
python -m evaluation.evaluate --live
```

For repeated live runs:

```bash
python -m evaluation.evaluate --live --repeats 3
```

The live run writes evaluation results and a summary under `evaluation/`. Read the generated files before reporting task success, tool-selection results, latency, token usage or failure cases.

A mock evaluation is useful for checking the evaluation harness, but **mock results are not evidence of real model performance**:

```bash
python -m evaluation.evaluate --mock
```

When presenting results, label mocked tests and live model evaluation separately. If a token metric is unavailable in the live response, report it as unavailable rather than estimating it.

## Suggested demonstration flow

- **Video 1 — Assignment 1:** student login, agent introduction, direct answer, fee tool, policy retrieval, memory across a new chat, sensitive-memory rejection, unsupported question, support-ticket draft/confirmation, run details, traces, evaluation and project structure.
- **Video 2 — Assignment 2:** both login tabs, student identity and own data, blocked cross-student request, pending record-change request, administrator rejection and approval, student-side status verification, ticket management, admin overview, records, policies, traces, audit log and security tests.

A detailed action-by-action voice-over script is provided in `VIDEO_SCRIPTS.md`.

## Limitations

- The student records and handbook are fictional demonstration data.
- Retrieval uses keyword/IDF scoring rather than semantic embeddings; synonyms and indirect wording may be missed.
- The live model's tool-choice reliability and response quality must be measured with the live evaluation command.
- The golden dataset is a limited test set and cannot represent every real student query.
- Name-based detection of another student can miss indirect descriptions; the tool layer still restricts student-data access to the authenticated student.
- Demo accounts are not a university identity-provider integration.
- This is an academic demonstration application, not a production university information system.

## Security checklist before publishing

Before pushing to a public GitHub repository, confirm that none of the following are staged or tracked:

- `.env` or any file containing an API key
- `data/agent.db`, other local databases or private runtime data
- `traces/` execution logs
- `testing/` demo credentials workbook
- private course handouts or copyrighted teaching materials
- screenshots or recordings showing passwords, OTPs, API keys or private student data

Useful checks:

```bash
git status --short
git ls-files | findstr /I ".env agent.db traces Demo_Test"
```

On macOS/Linux, use `grep -E` instead of `findstr`.

If a secret has ever been committed or shared, revoke/rotate it at the provider and create a new local key. Removing the file from the latest commit alone does not invalidate an exposed key.

## Author

**Student Support Agent — Agentic AI Engineering Assignments 1 and 2**
