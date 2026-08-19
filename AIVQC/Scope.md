# AIVQC — zakres i założenia projektu

> Żywy dokument projektu. Aktualizujemy go wraz z kolejnymi decyzjami, testami i wymaganiami.

## 1. Wizja

AI Visual Quality Controller (AIVQC) ma być uniwersalnym, dobrze zoptymalizowanym i łatwym do wdrożenia systemem wizyjnej kontroli jakości dla małych i średnich produkcji.

System powinien umożliwiać wykrywanie różnych rodzajów defektów niezależnie od branży i typu produktu. Ma ograniczać próg wejścia w uczenie maszynowe: użytkownik powinien móc zebrać dane, oznaczyć defekty, wytrenować i ocenić model, a następnie wdrożyć go na stanowisku produkcyjnym bez ręcznego przenoszenia wielu ustawień.

Projekt będzie składał się z dwóch osobnych aplikacji:

1. **AIVQC Trainer** — przygotowanie datasetów, trening, testowanie i eksport modeli.
2. **AIVQC Production** — wykonywanie inspekcji na stanowisku produkcyjnym.

## 2. Użytkownicy docelowi

- inżynier procesu lub jakości konfigurujący kontrolę,
- technik przygotowujący kamerę, dane i model,
- operator linii korzystający z gotowej inspekcji,
- małe i średnie zakłady bez własnego zespołu machine learning.

## 3. AIVQC Trainer

### 3.1. Cel

Szybkie i intuicyjne przeprowadzenie użytkownika przez cały proces: od zebrania zdjęć, przez przygotowanie datasetu i trening, aż do uzyskania gotowego, przetestowanego pakietu wdrożeniowego.

### 3.2. Zakres podstawowy

- tworzenie projektu dla konkretnego produktu i zadania inspekcyjnego,
- przechwytywanie zdjęć z kamery oraz import istniejących zdjęć i filmów,
- podgląd obrazu na żywo,
- konfiguracja kamery, między innymi rozdzielczości, ekspozycji, gain, balansu bieli, ostrości, FPS i obszaru zainteresowania — zależnie od możliwości urządzenia,
- zapis ustawień kamery razem z projektem i wyeksportowanym modelem,
- proste oznaczanie defektów zoptymalizowane pod dużą liczbę podobnych zdjęć,
- obsługa klas defektów i przykładów produktu prawidłowego,
- podział danych na zbiory treningowy, walidacyjny i testowy bez przecieku podobnych klatek między zbiorami,
- wybór spośród obsługiwanych architektur/modeli z czytelną informacją o wymaganiach i kompromisie szybkość–dokładność,
- konfiguracja treningu przy użyciu bezpiecznych ustawień domyślnych oraz trybu zaawansowanego,
- prezentacja postępu treningu i wyników,
- porównywanie kilku modeli na tym samym zbiorze testowym,
- test wydajności na docelowym sprzęcie lub na zdefiniowanym profilu sprzętowym,
- eksport kompletnego, wersjonowanego pakietu wdrożeniowego.

### 3.3. Wspomagane i automatyczne tworzenie datasetów

Automatyzację warto wdrażać etapami:

1. propagowanie oznaczeń na podobne lub kolejne klatki filmu,
2. wstępne oznaczanie przez istniejący model i zatwierdzanie/poprawianie przez użytkownika,
3. active learning — wskazywanie obrazów, których oznaczenie da największą wartość,
4. wykrywanie duplikatów, rozmytych zdjęć i problemów z ekspozycją,
5. opcjonalne wykorzystanie modeli typu foundation/segment-anything do szybkiego zaznaczania obiektów,
6. generowanie danych syntetycznych wyłącznie jako uzupełnienie danych rzeczywistych i z osobną walidacją.

Pełna automatyzacja bez kontroli człowieka nie jest założeniem początkowym. Błędne automatyczne etykiety mogą obniżyć jakość modelu w sposób trudny do zauważenia.

### 3.4. Ocena modelu

Trainer powinien raportować co najmniej:

- precision, recall oraz F1 dla każdej klasy defektu,
- mAP dla modeli detekcyjnych i IoU dla segmentacji,
- macierz pomyłek,
- liczbę false accept i false reject,
- wyniki dla każdej partii/serii danych, a nie tylko średnią globalną,
- czas inferencji, FPS, wykorzystanie CPU/GPU/RAM/VRAM i czas rozgrzewania modelu,
- galerię najtrudniejszych i błędnie sklasyfikowanych przykładów,
- sugerowany threshold, który użytkownik może zaakceptować lub zmienić.

### 3.5. Pakiet wdrożeniowy

Wyeksportowany pakiet powinien być samowystarczalny i zawierać:

- model oraz informację o jego formacie i wersji,
- identyfikator produktu/receptury,
- listę klas defektów,
- domyślne progi dla każdej klasy,
- wymagane przetwarzanie obrazu, rozmiar wejścia i normalizację,
- konfigurację kamery i ROI,
- metryki z walidacji oraz wymagania sprzętowe,
- wersję aplikacji/środowiska, w którym model został przygotowany,
- datę, autora i opcjonalne notatki wdrożeniowe,
- sumy kontrolne pozwalające wykryć uszkodzenie lub podmianę plików.

## 4. AIVQC Production

### 4.1. Cel

Stabilna, szybka i prosta obsługa inspekcji na produkcji z minimalną liczbą czynności wymaganych od operatora.

### 4.2. Zakres podstawowy

- wybór produktu/receptury, która automatycznie wskazuje właściwy model i ustawienia,
- możliwość ręcznego wyboru wersji modelu dla użytkownika z odpowiednimi uprawnieniami,
- automatyczne wczytanie konfiguracji kamery zapisanej w pakiecie wdrożeniowym,
- czytelna informacja, jeśli kamera nie obsługuje któregoś ustawienia lub ustawienie nie zostało zastosowane,
- podgląd obrazu i nałożonych detekcji,
- uruchamianie, zatrzymywanie i monitorowanie detekcji,
- osobny threshold dla każdej klasy defektu,
- możliwość zablokowania zmian thresholdów dla operatora,
- jednoznaczny wynik inspekcji: OK, NOK lub błąd/niepewny wynik,
- statystyki bieżącej zmiany i wybranego okresu,
- zapisywanie zdarzeń, błędów i zmian konfiguracji,
- opcjonalny zapis obrazów NOK oraz próbek OK zgodnie z polityką retencji,
- praca lokalna bez obowiązkowego dostępu do Internetu,
- bezpieczny powrót do ostatniej działającej wersji modelu.

### 4.3. Statystyki

- liczba i procent produktów OK/NOK,
- liczba defektów według klasy,
- false reject/false accept po ręcznym potwierdzeniu, jeśli taki proces będzie dostępny,
- wydajność: FPS, opóźnienie, czas pracy i liczba pominiętych klatek,
- trendy w czasie, według zmiany, partii i produktu,
- kondycja kamery i aplikacji,
- eksport danych do CSV; integracje z systemami zakładowymi w późniejszym etapie.

## 5. Model danych i pojęcia

- **Produkt** — fizyczny wyrób podlegający kontroli.
- **Receptura inspekcji** — połączenie produktu, wersji modelu, ustawień kamery, ROI, progów i reguł decyzji.
- **Model** — wersjonowany artefakt ML, który może być używany przez jedną lub więcej receptur.
- **Defekt** — klasa niezgodności wykrywana przez model.
- **Inspekcja** — pojedyncza ocena produktu lub obrazu zakończona wynikiem OK/NOK/błąd.

Produkt i model nie powinny być traktowane jako to samo. Jeden produkt może z czasem korzystać z kolejnych wersji modelu, a jedna receptura może zostać zmieniona bez ponownego treningu, na przykład przez korektę thresholdów.

## 6. Wymagania przekrojowe

- łatwa instalacja i konfiguracja na komputerach przemysłowych,
- wsparcie dla CPU, a opcjonalnie GPU i urządzeń edge,
- modularna obsługa różnych kamer i formatów modeli,
- działanie deterministyczne i odporność na chwilowy brak kamery lub uszkodzony pakiet,
- wersjonowanie konfiguracji i pełny audit log zmian,
- role użytkowników: operator, inżynier/administrator,
- możliwość tworzenia kopii zapasowej i przenoszenia receptur,
- wydajność mierzona na docelowym sprzęcie, nie tylko na komputerze treningowym,
- lokalne przechowywanie danych i jasna polityka retencji,
- architektura umożliwiająca późniejszą integrację z PLC, sygnalizacją, odrzutnikiem, MES lub API.

## 7. Proponowane etapy realizacji

### Etap 1 — MVP

- jeden obsługiwany typ zadania: detekcja obiektów,
- jeden główny format wdrożeniowy, preferencyjnie ONNX,
- tworzenie projektu, import/przechwytywanie obrazów i ręczne etykietowanie,
- trening lub uruchamianie procesu treningowego z poziomu Trainera,
- podstawowe metryki i benchmark,
- eksport/import pakietu wdrożeniowego,
- aplikacja Production z wyborem receptury, podglądem, thresholdami i podstawowymi statystykami,
- zapis ustawień kamery z kontrolą, czy zostały poprawnie zastosowane.

### Etap 2 — usprawnienie pracy z danymi

- pre-labeling, propagacja etykiet, active learning i kontrola jakości datasetu,
- porównywanie eksperymentów i wersji modeli,
- rozbudowane statystyki i raporty,
- zarządzanie użytkownikami oraz audit log.

### Etap 3 — integracje przemysłowe

- PLC/wejścia i wyjścia cyfrowe oraz sterowanie odrzutnikiem,
- integracja z MES/API,
- obsługa wielu kamer i stanowisk,
- centralne zarządzanie wdrożeniami, jeśli będzie potrzebne,
- dodatkowe zadania ML, na przykład segmentacja, klasyfikacja i wykrywanie anomalii.

## 8. Najważniejsze ryzyka

- zmienne oświetlenie, pozycja produktu i parametry optyki mogą wpływać na wynik bardziej niż wybór architektury modelu,
- losowy podział kolejnych klatek filmu może zawyżyć metryki przez przeciek danych,
- zbyt mała liczba przykładów rzadkich defektów utrudni wiarygodną ocenę,
- sam threshold nie rozwiąże problemu słabego lub niereprezentatywnego datasetu,
- konfiguracja kamery może nie być przenośna między różnymi modelami urządzeń,
- wynik laboratoryjny nie gwarantuje wymaganej wydajności i jakości na linii,
- automatyczne etykietowanie wymaga kontroli jakości przez człowieka,
- brak zdefiniowanej reakcji na błąd systemu może powodować przepuszczenie produktu bez ważnej inspekcji.

## 9. Kryteria sukcesu MVP

- nowy użytkownik potrafi utworzyć projekt, oznaczyć dane, wytrenować model i wyeksportować pakiet bez ręcznej edycji plików konfiguracyjnych,
- pakiet otwiera się w AIVQC Production i odtwarza model, klasy, preprocessing, ROI, progi oraz obsługiwane ustawienia kamery,
- aplikacja jasno sygnalizuje brak kamery, niezgodny model i brak możliwości zastosowania konfiguracji,
- benchmark raportuje powtarzalne wyniki jakości i wydajności,
- receptura oraz każda zmiana progów są wersjonowane lub rejestrowane,
- Production może pracować offline przez całą zmianę produkcyjną.

Docelowe wartości jakości, maksymalnego opóźnienia i FPS muszą zostać określone osobno dla konkretnego wdrożenia.

## 10. Poza zakresem MVP

- pełna automatyzacja etykietowania bez zatwierdzenia człowieka,
- jednoczesna obsługa wszystkich frameworków i modeli ML,
- chmurowe trenowanie i centralne zarządzanie flotą stanowisk,
- integracja ze wszystkimi sterownikami PLC i systemami MES,
- formalna certyfikacja dla branż regulowanych,
- automatyczne podejmowanie decyzji procesowych innych niż skonfigurowany wynik inspekcji.

## 11. Otwarte decyzje

- pierwszy docelowy typ produktu i zestaw defektów dla MVP,
- docelowy system operacyjny i minimalna specyfikacja sprzętu,
- pierwsze obsługiwane modele i sposób uruchamiania treningu,
- pierwsze obsługiwane kamery i biblioteka komunikacyjna,
- wymagany FPS, maksymalne opóźnienie oraz sposób wyzwalania inspekcji,
- czy oceniany będzie ciągły obraz, pojedynczy produkt po triggerze, czy oba tryby,
- reguła wyniku NOK przy wielu klasach i wielu detekcjach,
- sposób potwierdzania false accept/false reject,
- czas przechowywania obrazów i statystyk,
- format pakietu wdrożeniowego i strategia jego wersjonowania,
- technologia interfejsu obu aplikacji.

## 12. Rejestr zmian

- **2026-08-19 — wersja 0.1:** utworzenie dokumentu, podział na AIVQC Trainer i AIVQC Production, zdefiniowanie MVP, głównych wymagań, ryzyk i otwartych decyzji.
