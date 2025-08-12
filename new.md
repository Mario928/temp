# Status Update & Development Plan: Chat History Export Feature

## High-Level Summary (For Stand-Up)

Good morning, team.

For the chat history task, my focus last week and yesterday was on the initial setup and analysis phase.  
-   Setting up and configuring the local development environment.
-    Gaining a understanding of the feature requirements.
-    Based on the requirements, I began mapping out the development scope and initial roadmap. This involved a dive into the codebase, debugging existing functionalities to understand the data flow.

Based on that investigation, here is a high-level overview of my analysis and the proposed impleemtnation 



**Key Findings from Analysis:**
Information such as the specific models used per chat, message counts, user feedback, and timestamps is being stored within the `chat` table's JSON data structure.

This is a good  bcz it means we can deliver a powerful export tool **without undertaking complex and high-risk database schema migrations** for the initial version. The challenge isn't a lack of data, but the need for a dedicated data orchestration and processing pipeline. which i will wokr on for this week as i am back full capcaociity

**My Development Plan for the Upcoming Week:**

Based on this analysis, I've formulated a two-phase development plan. My focus this week will be on **Phase 1: The Core Export MVP**.

The plan involves architecting a new, secure data pipeline that will:
1.  **Enhance our API** by introducing a new, protected endpoint within the existing `/groups` router. This endpoint will be responsible for handling export requests for specific user groups.
2.  **Implement an efficient data aggregation layer** in the backend. This will involve a new database access function to gather all chat data for multiple users in a single, optimized query.
3.  **Develop a server-side processing engine** using the `pandas` library. This engine will parse the raw JSON chat data, calculate the required metadata fields (like message count and time spent), and transform it into a structured Excel format. This is done on the server to ensure high performance and not overload the user's browser.
4.  **Integrate a minimal UI component** on the frontend—a simple "Download" button on the admin group panel—that will trigger this entire backend process and stream the generated report securely to the professor.



## Detailed Technical Plan (For Reference)

### 1. Codebase Analysis & Findings

After a two-day deep dive into the NAGA-open-webui repository, I have a clear understanding of the architecture and the data flow relevant to the chat history export feature.

*   **System Architecture:**
    *   **Frontend:** SvelteKit Single Page Application (SPA).
    *   **Backend:** FastAPI (Python) serving a REST API.
    *   **Database:** SQLAlchemy ORM connected to a SQL database.

*   **Chat Data Storage:**
    *   Chats are stored in the `chat` table, defined in `backend/open_webui/models/chats.py`.
    *   The storage model is a hybrid:
        *   **Structured SQL Columns:** `id`, `user_id`, `title`, `created_at`.
        *   **Flexible JSON Column:** A column named `chat` of type `JSON` stores the entire conversation object. This includes the list of messages, models used, and other rich metadata.

*   **Key Finding:**
    *   All data required for the export is **already being collected**. The `chat` JSON object contains everything we need to calculate the required metrics.
    *   **Conclusion:** No database schema changes are required for the MVP. The primary task is to fetch, parse, and format this existing data.

### 2. Development Plan: A Phased Approach

#### Phase 1: MVP - Bulk Export Functionality (This Week's Goal)

The initial goal is to provide professors with the raw data they need in a usable format. Filtering and analysis will be done manually by the user in Excel.

**User Flow:**
1.  Professor navigates to the Admin > Groups page.
2.  Clicks a "Download Report" icon next to a specific group.
3.  An Excel file containing all chat metadata for every student in that group is downloaded.

**Technical Implementation:**

1.  **Dependencies (`backend/requirements.txt`):**
    *   Add `pandas` and `openpyxl`.
2.  **Backend (`backend/open_webui/models/chats.py`):**
    *   **Action:** Create a new function: `get_chats_by_user_ids(self, user_ids: list[str])`.
    *   **Justification:** To avoid the "N+1 query problem" by getting all required chats in a single, efficient database call.
3.  **Backend (`backend/open_webui/routers/groups.py`):**
    *   **Action:** Create one new API endpoint: `GET /api/groups/{group_id}/chats/export`.
    *   **Logic:** Authorize -> Get Group Members -> Get Chat Data -> Process Data -> Generate Excel -> Return File Stream.
4.  **Frontend (`src/lib/components/admin/Users/Groups.svelte`):**
    *   **Action:** Add a new icon button to each row in the groups table.
5.  **Frontend (`src/lib/apis/groups/index.ts`):**
    *   **Action:** Create a new function: `exportGroupChats(token, groupId)`.
    *   **Logic:** Calls the new endpoint, receives the response as a `blob`, and uses `file-saver` to trigger the download.

---



















#### Phase 2: Future Enhancement - In-App Filtering UI

This phase builds on the MVP to provide a more interactive and user-friendly experience.

**User Flow:**
1.  Professor navigates to the group page and sees new filter controls.
2.  Applying a filter triggers an API call to get a filtered list of students, which updates the UI dynamically.
3.  The "Download" button exports the data for only the filtered results.

**Technical Implementation (High-Level):**

*   **Two New API Endpoints:**
    1.  `POST /api/groups/{group_id}/chats/filter`: Takes filter criteria, returns a **JSON list** for the UI.
    2.  `POST /api/groups/{group_id}/chats/export`: Takes the same filter criteria, returns an **Excel file stream**.
*   **Backend Logic:** The filtering logic will be implemented in Python on the backend.
*   **Frontend Work:** Requires significant UI development for filter components and dynamic list rendering.

---

## Appendix: Data Source Reference

This section details exactly where the data for the export comes from. No modifications are needed to access this data.

### Chat Database Table (`chat`)

This is the SQLAlchemy model definition from `backend/open_webui/models/chats.py`. It shows the structured columns.

```python
class Chat(Base):
    __tablename__ = "chat"

    id = Column(String, primary_key=True)
    user_id = Column(String)
    title = Column(Text)
    chat = Column(JSON)  # <-- All the rich metadata is inside this field

    created_at = Column(BigInteger)
    updated_at = Column(BigInteger)

    share_id = Column(Text, unique=True, nullable=True)
    archived = Column(Boolean, default=False)
    pinned = Column(Boolean, default=False, nullable=True)

    meta = Column(JSON, server_default="{}")
    folder_id = Column(Text, nullable=True)
```

### Chat JSON Structure (Example from the `chat` column)

This is a simplified example of the data stored inside the `chat` JSON column for a single chat session.

```json
{
  "title": "History of Artificial Intelligence",
  "models": [
    "gemma-7b-it"
  ],
  "messages": [
    {
      "id": "message-1",
      "role": "user",
      "content": "What was the Turing Test?",
      "timestamp": 1678886400
    },
    {
      "id": "message-2",
      "role": "assistant",
      "content": "The Turing Test, developed by Alan Turing...",
      "feedback": "thumbsUp",
      "timestamp": 1678886405
    }
  ],
  "timestamp": 1678886400
}
```

### Available Metadata (Direct Access vs. Calculated)

#### Directly Accessible Data:
*   **NetID:** From the `user_id` column.
*   **Chat Title:** From the `title` column.
*   **Chat Date:** From the `created_at` column.
*   **Model Used:** From the `models` array within the JSON.
*   **User Feedback:** From the `feedback` field on each message within the JSON.

#### Calculated Metadata (We will compute these in the backend):
*   **Message Count:** By counting the number of elements in the `messages` array.
*   **Time Spent on Chat:** By calculating the difference between the `timestamp` of the last message and the first message.
