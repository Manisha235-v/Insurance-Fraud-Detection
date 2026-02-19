ddocument.getElementById('fraudForm').addEventListener('submit', function(e) {
    e.preventDefault();

    // Get values from form
    let age = document.getElementById('age').value;
    let claimAmount = document.getElementById('claim_amount').value;
    let incidentType = document.getElementById('incident_type').value;
    let policeReport = document.getElementById('police_report').value;
    let witnesses = document.getElementById('witnesses').value;
    let incidentSeverity = document.getElementById('incident_severity').value;
    let vehicleClaim = document.getElementById('vehicle_claim').value;
    let income = document.getElementById('income').value;
    let previousClaims = document.getElementById('previous_claims').value;
    let location = document.getElementById('location').value;
    let medicalReport = document.getElementById('medical_report').value;
    let accidentPhoto = document.getElementById('accident_photo').files[0];

    // Fraud detection logic
    let fraudScore = 0;
    let reasons = [];

    if (claimAmount > 20000) { fraudScore += 20; reasons.push('High claim amount'); }
    if (policeReport == 0) { fraudScore += 15; reasons.push('No police report filed'); }
    if (incidentType == 'Fire') { fraudScore += 15; reasons.push('Fire incident is high risk'); }
    if (incidentType == 'Theft') { fraudScore += 10; reasons.push('Theft incident detected'); }
    if (age < 25) { fraudScore += 10; reasons.push('Claimant is under 25'); }
    if (witnesses == 0) { fraudScore += 10; reasons.push('No witnesses present'); }
    if (incidentSeverity == 'Total Loss') { fraudScore += 15; reasons.push('Total loss claimed'); }
    if (vehicleClaim > 15000) { fraudScore += 10; reasons.push('High vehicle claim'); }
    if (previousClaims > 2) { fraudScore += 20; reasons.push('Multiple previous claims'); }
    if (income < 20000 && claimAmount > 10000) { fraudScore += 15; reasons.push('Low income but high claim'); }
    if (location == 'metro') { fraudScore += 5; reasons.push('Metro city high fraud zone'); }
    if (medicalReport == 0) { fraudScore += 15; reasons.push('No medical report submitted'); }
    if (!accidentPhoto) { fraudScore += 10; reasons.push('No accident photo uploaded'); }

    // Cap score at 100
    if (fraudScore > 100) fraudScore = 100;

    // Show result
    let result = document.getElementById('result');
    result.style.display = 'block';

    let reasonsList = reasons.map(r => `<li>⚠️ ${r}</li>`).join('');

    if (fraudScore >= 50) {
        result.innerHTML = `
        <h3 style="color:red;">⚠️ FRAUDULENT CLAIM</h3>
        <p><b>Fraud Score: ${fraudScore}%</b></p>
        <p>Risk Level: ${fraudScore >= 75 ? '🔴 High Risk' : '🟠 Medium Risk'}</p>
        <p>This claim has been flagged as suspicious!</p>
        <hr>
        <h4>Reasons:</h4>
        <ul style="text-align:left;">${reasonsList}</ul>`;
        result.style.background = '#ffe0e0';
        result.style.border = '2px solid red';
    } else {
        result.innerHTML = `
        <h3 style="color:green;">✅ LEGITIMATE CLAIM</h3>
        <p><b>Fraud Score: ${fraudScore}%</b></p>
        <p>Risk Level: 🟢 Low Risk</p>
        <p>This claim looks legitimate!</p>
        <hr>
        <h4>Reasons:</h4>
        <ul style="text-align:left;">${reasonsList}</ul>`;
        result.style.background = '#e0ffe0';
        result.style.border = '2px solid green';
    }

    // Scroll to result
    result.scrollIntoView({ behavior: 'smooth' });
});
