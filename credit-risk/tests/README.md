# NeMo-PD: Mortgage Probability of Default

End-to-end, production-minded PD@24m model on Freddie Mac loan-level data.

## Quickstart
```bash
# macOS/Linux
make setup
make abt      # Step 2 will build ABT
make features # Step 4
make train    # Step 5
make evaluate # Step 5–6
make dashboard
make serve
