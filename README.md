# Object Detection and Phone Counting with YOLO

Questo progetto implementa il rilevamento di oggetti e il conteggio dei telefoni utilizzando il modello YOLO (You Only Look Once) e il dataset di base di YOLO. Utilizzeremo OpenCV per l'elaborazione delle immagini e per eseguire il rilevamento degli oggetti in tempo reale.

## Requisiti

Per avviare il progetto, è sufficiente installare le librerie necessarie contenute nel file `requirements.txt`. Esegui i seguenti comandi per preparare l'ambiente:

1. Clona questo repository:
    ```bash
    git clone https://github.com/tuo-utente/oggetto-detect-telefono.git
    cd oggetto-detect-telefono
    ```

2. Crea un ambiente virtuale e attivalo:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Su Windows, usa venv\Scripts\activate
    ```

3. Installa le librerie richieste:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset YOLO

Il dataset di base di YOLO per il rilevamento degli oggetti viene utilizzato per identificare e contare i telefoni nelle immagini. Per maggiori dettagli sul dataset, visita i seguenti link:

- [Darknet YOLO GitHub Repository](https://github.com/AlexeyAB/darknet?tab=readme-ov-file)
- [Coco Names File](https://github.com/AlexeyAB/darknet/blob/master/data/coco.names)

## Come Avviare

1. Assicurati di aver scaricato i pesi di YOLO pre-addestrati. Puoi farlo dal link ufficiale di YOLO o dal tuo file di configurazione personalizzato.
   
2. Avvia il file `main.py` per eseguire il rilevamento degli oggetti e il conteggio dei telefoni. Usa il comando:

    ```bash
    python main.py
    ```

3. Lo script caricherà il modello YOLO, eseguirà il rilevamento e stamperà il conteggio dei telefoni rilevati in tempo reale.

## Struttura del Progetto
```
oggetto-detect-telefono/
│
├── main.py               # Script principale per il rilevamento e il conteggio
|── utils.py              # Contiene le variabili di configurazione
├── requirements.txt      # File con le dipendenze Python
├── yolo_model/           # Contiene i pesi di YOLO (ad esempio, yolov4.weights)
│   ├── yolov4.weights    # Pesi pre-addestrati di YOLO
│   ├── yolov4.cfg        # Configurazione di YOLO
│   └── coco.names        # File delle classi per YOLO (ad esempio, coco.names)
├── README.md             # Questo file
└── LICENSE               # File con la licenza del progetto
```

## Licenza

Distribuito sotto la licenza MIT. Vedi il file [LICENSE](LICENSE) per ulteriori dettagli.
