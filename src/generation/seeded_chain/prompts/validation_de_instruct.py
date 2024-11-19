from langchain_core.prompts import ChatPromptTemplate

system_prompt = """
    Deine Aufgabe ist es, das folgende Gesprächsskript zu validieren und zu korrigieren.
    Stelle sicher, dass die Ausgabe ein gültiges JSON-Objekt ist, das strikt der vorgegebenen Struktur entspricht.
    Das JSON-Objekt muss:
    - Doppelte Anführungszeichen für alle Schlüssel und Zeichenkettenwerte verwenden.
    - Leere oder fehlende Werte als 'null' (ohne Anführungszeichen) darstellen.
    - Keinen zusätzlichen Text, keine Kommentare oder Erklärungen außerhalb des JSON-Formats enthalten.

    Stelle außerdem sicher, dass das Gespräch:
    - Professionell, höflich, informativ und relevant für Bankdienstleistungen ist.
    - Auf Deutsch verfasst ist, mit korrekter Grammatik und Rechtschreibung.

    Gib ausschliesslich das korrigierte JSON-Objekt zurück und stelle sicher, dass es allen oben genannten Richtlinien folgt.
    Füge keine zusätzlichen Kommentare oder Erklärungen hinzu.
"""

user_prompt = """
    Validieren und korrigieren Sie das folgende generierte Gesprächsskript.

    {script_json}

    Stellen Sie sicher, dass:
    - Die Personen als "Agent" und "Client" bezeichnet werden.
    - Der Name des Kunden, falls möglich, in der Rückrufnotiz enthalten ist.
    - Der Inhalt in korrektem Deutsch verfasst ist, mit richtiger Grammatik und Rechtschreibung.
    - Der Output ein gültiges JSON-Objekt ist, das strikt der vorgegebenen JSON-Struktur folgt.
    Geben Sie nur das korrigierte JSON-Objekt zurück, ohne zusätzlichen Text oder Kommentare.
    """

validation_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", user_prompt)])
