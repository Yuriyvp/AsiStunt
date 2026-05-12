# Eva Novak — kratak pregled za vlasnika

**AI glasovna asistentica za Dogma Nekretnine**
Status: MVP (rana radna verzija)

---

## Što ovo jest

Eva Novak je AI glasovna asistentica koja prima prve dolazne upite klijenata na pet jezika (hrvatski, engleski, ruski, njemački, talijanski), vodi kratak razgovor o tome što klijent traži, i predaje kvalificirani kontakt vašem agentu — najčešće Lidi. Ne predstavlja se kao osoba, ne daje tržišne, pravne ni porezne savjete, i ne izmišlja nekretnine. To su namjerne granice — štite agenciju.

---

## Koliko je spremna danas

**Trenutna spremnost: oko 60–70%.** Ovo je MVP — minimalna radna verzija namijenjena demonstraciji i pilot fazi, ne masovnoj produkciji.

U ovoj fazi su **očekivane**:

- povremene greške u prepoznavanju govora (osobito na hrvatskim riječima sa specifičnim akcentima)
- rijetka krivo razumijevanja konteksta razgovora
- ponekad spor odgovor pri velikoj povijesti razgovora

Ovo nije neuspjeh sustava — ovo je normalno stanje MVP-a koji koristi opremu koju imamo trenutno. Sve se rješava u sljedećim fazama.

---

## Što se može poboljšati s jačim hardverom

Trenutni sustav radi na jednoj GPU kartici (24 GB). S jačim hardverom (npr. profesionalna GPU od 48–96 GB) odmah dobivamo:

- **Točnije prepoznavanje govora** — manje grešaka, posebno na hrvatskom
- **Pametniji "mozak"** (veći jezični model) — bolje razumijevanje konteksta, prirodniji odgovori
- **Bolja kvaliteta glasa** — još bliža ljudskom govoru

Realna procjena: **10–20% kvalitetnije korisničko iskustvo** na svim razinama, samo nadogradnjom hardvera, bez promjene logike sustava.

---

## Što je trenutno, a što je temelj za budućnost

Trenutno: **glasovna jezgra**. Eva razgovara, kvalificira, sažima razgovor u tekstualnu datoteku, i predaje informacije agentu. Sve ostalo agent radi ručno.

Ova jezgra je **temelj** — sve sljedeće faze samo dograđuju na nju:

### Faza 2 — IPBX (telefonija)

Trenutno Eva radi preko web-widgeta i računala. Spajanjem na IPBX (telefonsku centralu) Eva preuzima i klasične telefonske pozive — kupac zove broj Dogme, Eva se javlja kao prva linija, kvalificira poziv, prosljeđuje agentu ili snima poruku. Ovo zahtijeva dodatno testiranje jer je telefonski zvuk slabije kvalitete od mikrofonskog.

### Faza 3 — RAG (pristup stvarnoj bazi nekretnina)

**Što je RAG, jednostavno:** danas Eva ima u glavi samo opće znanje o Rijeci i o Dogmi. Ne zna stvarne nekretnine iz vašeg portfolija. RAG ("retrieval-augmented generation") znači da prije svakog odgovora Eva ode u vašu bazu nekretnina, pronađe konkretnu nekretninu koja odgovara upitu, i odgovori s **pravim** podacima — pravom adresom, pravom cijenom, pravim kvadratima.

Bez RAG-a: *"Imam tri demo oglasa, za stvarne nekretnine ću vas spojiti s agentom."*
S RAG-om: *"Imamo dvosoban stan na Pećinama, 65 m², 235.000 €, pogled na more, slobodan za pregled u srijedu."*

Tehnički je RAG zahtjevniji projekt — treba povezati Evu sa sustavom u kojem se vode oglasi, postaviti automatsko osvježavanje, i osigurati da Eva ne odgovara na temelju zastarjelih oglasa. Vrijedi truda jer drastično mijenja kvalitetu razgovora.

### Faza 4 — CRM i e-mail integracija

Trenutno se sažetci razgovora spremaju kao datoteke na disku. Sljedeći korak je automatsko ubacivanje kvalificiranih kontakata u Dogmin CRM (kojigod sustav koristite za vođenje klijenata) i automatsko slanje e-maila agentu nakon završetka poziva, s priloženim sažetkom i transkriptom. Agent jutrom otvori inbox i ima sve.

---

## Sažetak

- **Danas:** glasovna asistentica koja kvalificira dolazne upite na 5 jezika i predaje kontakt agentu — radna, ali MVP.
- **Iduća razina hardvera:** 10–20% bolja kvaliteta na svim razinama.
- **Iduće faze:** telefonija (IPBX) → stvarne nekretnine (RAG) → CRM i e-mail.

Eva sada nije zamjena za agenta. Eva je način da niti jedan dolazni upit — neovisno o satu, jeziku ili kanalu — ne ostane bez odgovora prije nego što ga vaš agent može preuzeti.
