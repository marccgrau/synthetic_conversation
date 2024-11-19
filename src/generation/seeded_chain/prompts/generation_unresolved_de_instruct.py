from langchain_core.prompts import ChatPromptTemplate

system_prompt = """
    Deine Aufgabe ist es, ein Gesprächsskript für einen Kundenserviceanruf zwischen einem Kunden und einem Finanzdienstleister zu erstellen.
    Das Gesprächsskript soll auf Deutsch geschrieben sein und als gültiges JSON-Objekt ausgegeben werden.
    Stelle sicher, dass:
    - Das JSON strikt der vorgegebenen Struktur entspricht.
    - 'null' (nicht 'None') für alle leeren oder fehlenden Werte verwendet wird.
    - Kein zusätzlicher Text, keine Kommentare oder Erklärungen außerhalb des JSON-Formats enthalten sind.
    - Alle JSON-Schlüssel und -Werte in doppelte Anführungszeichen gesetzt werden, außer 'null'.
"""

user_prompt = """
    Erstelle ein Gesprächsskript für einen Kundenserviceanruf zwischen einem Kunden und einem Kundendienstmitarbeiter einer Schweizer Bank.
    Verwende Kantonalbank als Bankname.
    Das Skript soll auf Deutsch, professionell, höflich und informativ sein.

    - **Teilnehmer**: Agent (Kundendienstmitarbeiter), Kunde (Kunde).
    - **Struktur**:
        - Begrüßung
        - Beschreibung des Problems/der Frage des Kunden
        - Kundenidentifikation (spezifische Kontodetails oder Kundendetails)
        - Detaillierte Beschreibung der Anfrage
        - Hinweis, dass das Problem Nachverfolgung erfordert oder während des Anrufs nicht gelöst werden kann
        - Der Agent bietet weitere Hilfe an, der Kunde lehnt ab, der Agent bedankt sich beim Kunden und beendet das Gespräch.

    Verwende die folgende Struktur als Beispiel, um ein neues Skript und eine Rückrufnotiz mit anderem Inhalt zu erstellen:

    {structure_json}

    Die Struktur muss nicht im Output enthalten sein. Sie dient nur als Referenz.

    Zusätzlich gibt es ein Beispiel für ein Gespräch:

    {example_json}

    **Neues Inhaltsthema**:
    {topic}

    Stelle sicher, dass der neue Inhalt die gleiche Struktur beibehält, aber dem neuen Thema folgt.
    Das Skript sollte im Wechselsprechformat und einzigartig sein.

    **Anforderungen an die Rückrufnotiz**:
    - person_number: 123.456.789.0
    - phone_number: 079 111 11 11
    - message: Zusammenfassung des Gesprächsskripts, die das ungelöste Problem und den Bedarf an weiterer Nachverfolgung angibt (nur für interne Dokumentation)
    - resolved_items: null
    - action_items: {topic}
    - wants_callback: true
    - phone_private: 0799111010
    - remark: null

    Das gesamte Gespräch muss auf Deutsch mit korrekter Grammatik und Rechtschreibung verfasst sein.
    Erstelle das Gespräch so, dass es strikt dem vorgegebenen neuen Thema und der Struktur folgt.
    Gib nur das JSON-Objekt mit dem Gespräch und der Rückrufnotiz zurück.
    Füge keinen zusätzlichen Text oder Kommentare hinzu.
    """  # noqa

generation_unresolved_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("user", user_prompt)])
