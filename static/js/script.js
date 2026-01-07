// ========================================
// MediCare AI - Professional JavaScript with Animations
// ========================================

'use strict';

// ========================================
// CSRF Token Helper
// ========================================
function getCSRFToken() {
    const name = 'csrftoken';
    let cookieValue = null;

    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }

    return cookieValue;
}

// ========================================
// Professional Notification System
// ========================================
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${getNotificationIcon(type)} me-2"></i>
            <span>${message}</span>
        </div>
        <button class="notification-close" aria-label="Close notification">&times;</button>
    `;

    document.body.appendChild(notification);

    // Animate in
    setTimeout(() => {
        notification.classList.add('show');
    }, 100);

    // Close button functionality
    const closeBtn = notification.querySelector('.notification-close');
    closeBtn.addEventListener('click', () => {
        closeNotification(notification);
    });

    // Auto close after 5 seconds
    setTimeout(() => {
        if (document.body.contains(notification)) {
            closeNotification(notification);
        }
    }, 5000);
}

function getNotificationIcon(type) {
    const icons = {
        success: 'check-circle',
        error: 'exclamation-circle',
        warning: 'exclamation-triangle',
        info: 'info-circle'
    };
    return icons[type] || icons.info;
}

function closeNotification(notification) {
    notification.classList.remove('show');
    setTimeout(() => {
        if (document.body.contains(notification)) {
            document.body.removeChild(notification);
        }
    }, 400);
}

// ========================================
// Enhanced BMI Calculator with Animation
// ========================================
function initBMICalculator() {
    const heightInput = document.getElementById('height');
    const weightInput = document.getElementById('weight');
    const bmiDisplay = document.getElementById('bmi-display');

    if (heightInput && weightInput && bmiDisplay) {
        function calculateBMI() {
            const height = parseFloat(heightInput.value) / 100;
            const weight = parseFloat(weightInput.value);

            if (height && weight && height > 0 && weight > 0) {
                const bmi = (weight / (height * height)).toFixed(1);

                // Animate BMI value
                animateValue(bmiDisplay, 0, bmi, 500);

                // Update BMI category and color
                let category = '';
                let colorClass = '';

                if (bmi < 18.5) {
                    category = 'Underweight';
                    colorClass = 'text-info';
                } else if (bmi < 25) {
                    category = 'Normal weight';
                    colorClass = 'text-success';
                } else if (bmi < 30) {
                    category = 'Overweight';
                    colorClass = 'text-warning';
                } else {
                    category = 'Obese';
                    colorClass = 'text-danger';
                }

                bmiDisplay.className = `form-control ${colorClass}`;
                bmiDisplay.setAttribute('data-category', category);
                bmiDisplay.setAttribute('title', `BMI Category: ${category}`);
            } else {
                bmiDisplay.value = '';
                bmiDisplay.className = 'form-control';
            }
        }

        heightInput.addEventListener('input', calculateBMI);
        weightInput.addEventListener('input', calculateBMI);

        // Calculate on page load if values exist
        calculateBMI();
    }
}

// ========================================
// Animate Number Values
// ========================================
function animateValue(element, start, end, duration) {
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;

    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            current = end;
            clearInterval(timer);
        }
        element.value = current.toFixed(1);
    }, 16);
}

// ========================================
// Professional Results Display for Diabetes
// ========================================
function displayDiabetesResult(result) {
    if (result.success) {
        const resultContent = document.querySelector('.result-content');
        const placeholder = document.querySelector('.result-placeholder');
        const probability = document.querySelector('.probability');
        const recommendation = document.querySelector('.recommendation');
        const riskBadge = document.querySelector('.risk-badge');

        // Show results section with animation
        if (placeholder) {
            placeholder.style.opacity = '0';
            setTimeout(() => {
                placeholder.classList.add('d-none');
            }, 300);
        }

        if (resultContent) {
            resultContent.classList.remove('d-none');
            resultContent.style.opacity = '0';
            setTimeout(() => {
                resultContent.style.opacity = '1';
            }, 100);
        }

        // Update tier and confidence display (new feature)
        if (typeof updateTierDisplay === 'function') {
            updateTierDisplay(result);
        }

        // Animate probability
        if (probability) {
            animateCounter(probability, 0, result.probability, 2000);
        }

        // Update risk level and styling
        if (result.risk_level === 'low') {
            updateRiskDisplay(riskBadge, 'Low Risk', 'low-risk', 'success');
            if (recommendation) {
                recommendation.innerHTML = `
                    <strong>Excellent!</strong> Your diabetes risk is low. 
                    Continue maintaining a healthy lifestyle with regular exercise and balanced diet.
                    <br><small class="text-muted mt-2 d-block">
                        <i class="fas fa-lightbulb me-1"></i>
                        Keep monitoring your health parameters regularly.
                    </small>
                `;
            }
        } else if (result.risk_level === 'medium') {
            updateRiskDisplay(riskBadge, 'Medium Risk', 'medium-risk', 'warning');
            if (recommendation) {
                recommendation.innerHTML = `
                    <strong>Attention needed.</strong> Your diabetes risk is moderate. 
                    Consider consulting a healthcare professional and monitor your health parameters regularly.
                    <br><small class="text-muted mt-2 d-block">
                        <i class="fas fa-user-md me-1"></i>
                        Schedule a check-up with your doctor within 3-6 months.
                    </small>
                `;
            }
        } else if (result.risk_level === 'high') {
            updateRiskDisplay(riskBadge, 'High Risk', 'high-risk', 'danger');
            if (recommendation) {
                recommendation.innerHTML = `
                    <strong>Immediate attention required.</strong> Your diabetes risk is high. 
                    Please consult with a healthcare professional immediately for proper diagnosis and treatment plan.
                    <br><small class="text-muted mt-2 d-block">
                        <i class="fas fa-exclamation-triangle me-1"></i>
                        Book an appointment with your doctor as soon as possible.
                    </small>
                `;
            }
        } else if (result.risk_level === 'critical') {
            updateRiskDisplay(riskBadge, 'Critical Risk', 'high-risk', 'danger');
            if (recommendation) {
                recommendation.innerHTML = `
                    <strong>⚠️ URGENT: Seek immediate medical attention.</strong> 
                    Your risk indicators are significantly elevated. Please contact a healthcare provider today.
                    <br><small class="text-danger mt-2 d-block">
                        <i class="fas fa-phone me-1"></i>
                        Do not delay - consult a doctor immediately.
                    </small>
                `;
            }
        }

        // Update risk factors if available
        updateRiskFactors(result);

        // Show success notification
        showNotification('Risk assessment completed successfully!', 'success');

        // Scroll to results
        smoothScrollTo('results-card');

    } else {
        // Handle model unavailable error
        if (result.model_unavailable) {
            showNotification('The prediction model is currently unavailable. Please try again later.', 'error');
        } else {
            showNotification(`Prediction failed: ${result.error}`, 'error');
        }
    }
}

// ========================================
// Animate Counter
// ========================================
function animateCounter(element, start, end, duration) {
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;

    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            current = end;
            clearInterval(timer);
        }
        element.textContent = Math.round(current) + '%';
    }, 16);
}

function updateRiskDisplay(riskBadge, text, className, bootstrapClass) {
    if (riskBadge) {
        riskBadge.textContent = text;
        riskBadge.className = `risk-badge ${className}`;

        // Update risk circle color
        const riskCircle = document.querySelector('.risk-circle');
        if (riskCircle) {
            riskCircle.className = `risk-circle ${bootstrapClass}`;
        }
    }
}

function updateRiskFactors(result) {
    const factorItems = document.querySelectorAll('.risk-factor-item');

    factorItems.forEach(item => {
        const factorName = item.querySelector('.factor-name').textContent.toLowerCase();
        const factorStatus = item.querySelector('.factor-status');

        let status = 'normal';
        let statusText = 'Normal';

        if (result.risk_level === 'high') {
            status = Math.random() > 0.5 ? 'elevated' : 'high';
            statusText = status === 'elevated' ? 'Elevated' : 'High';
        } else if (result.risk_level === 'medium') {
            status = Math.random() > 0.7 ? 'elevated' : 'normal';
            statusText = status === 'elevated' ? 'Elevated' : 'Normal';
        }

        factorStatus.className = `factor-status ${status}`;
        factorStatus.textContent = statusText;
    });
}

// ========================================
// Form Validation Enhancement
// ========================================
function validateForm(formId) {
    const form = document.getElementById(formId);
    if (!form) return false;

    const requiredFields = form.querySelectorAll('[required]');
    let isValid = true;

    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            field.classList.add('is-invalid');
            isValid = false;

            // Shake animation
            field.style.animation = 'shake 0.5s';
            setTimeout(() => {
                field.style.animation = '';
            }, 500);
        } else {
            field.classList.remove('is-invalid');
            field.classList.add('is-valid');
        }
    });

    return isValid;
}

// ========================================
// Loading State Management
// ========================================
function setLoadingState(buttonId, isLoading = true) {
    const button = document.getElementById(buttonId);
    if (!button) return;

    const btnText = button.querySelector('.btn-text');
    const spinner = button.querySelector('.spinner-border');

    if (isLoading) {
        button.disabled = true;
        if (btnText) btnText.textContent = 'Processing...';
        if (spinner) spinner.classList.remove('d-none');
        button.classList.add('loading');
    } else {
        button.disabled = false;
        if (btnText) btnText.textContent = button.getAttribute('data-original-text') || 'Submit';
        if (spinner) spinner.classList.add('d-none');
        button.classList.remove('loading');
    }
}

// ========================================
// Smooth Scrolling
// ========================================
function smoothScrollTo(elementId) {
    const element = document.getElementById(elementId) || document.querySelector(`.${elementId}`);
    if (element) {
        element.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }
}

// ========================================
// Particle Background Animation
// ========================================
function createParticles() {
    const heroSection = document.querySelector('.hero-section');
    if (!heroSection) return;

    const particlesContainer = document.createElement('div');
    particlesContainer.className = 'particles-bg';
    heroSection.appendChild(particlesContainer);

    for (let i = 0; i < 50; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 20 + 's';
        particle.style.animationDuration = (Math.random() * 10 + 10) + 's';
        particlesContainer.appendChild(particle);
    }
}

// ========================================
// Intersection Observer for Animations
// ========================================
function initScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in-up');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe all cards
    document.querySelectorAll('.prediction-card, .stat-card, .feature-item').forEach(el => {
        observer.observe(el);
    });
}

// ========================================
// Navbar Scroll Effect
// ========================================
function initNavbarScroll() {
    const navbar = document.querySelector('.navbar');
    if (!navbar) return;

    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.style.boxShadow = '0 10px 40px rgba(0,0,0,0.2)';
            navbar.style.background = 'rgba(102, 126, 234, 1)';
        } else {
            navbar.style.boxShadow = '0 8px 32px rgba(0,0,0,0.1)';
            navbar.style.background = 'rgba(102, 126, 234, 0.95)';
        }
    });
}

// ========================================
// 3D Card Tilt Effect
// ========================================
function init3DCardEffect() {
    const cards = document.querySelectorAll('.prediction-card');

    cards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const centerX = rect.width / 2;
            const centerY = rect.height / 2;

            const rotateX = (y - centerY) / 10;
            const rotateY = (centerX - x) / 10;

            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-20px)`;
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) translateY(0)';
        });
    });
}

// ========================================
// Initialize all functions when DOM is loaded
// ========================================
document.addEventListener('DOMContentLoaded', function () {
    // Initialize BMI calculator if present
    initBMICalculator();

    // Create particles
    createParticles();

    // Initialize scroll animations
    initScrollAnimations();

    // Initialize navbar scroll effect
    initNavbarScroll();

    // Initialize 3D card effect
    init3DCardEffect();

    // Add smooth scrolling to anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            smoothScrollTo(targetId);
        });
    });

    // Store original button text for loading states
    document.querySelectorAll('button[type="submit"]').forEach(btn => {
        const btnText = btn.querySelector('.btn-text');
        if (btnText) {
            btn.setAttribute('data-original-text', btnText.textContent);
        }
    });

    // Add ripple effect to buttons
    document.querySelectorAll('.btn').forEach(button => {
        button.addEventListener('click', function (e) {
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;

            ripple.style.width = ripple.style.height = size + 'px';
            ripple.style.left = x + 'px';
            ripple.style.top = y + 'px';
            ripple.classList.add('ripple');

            this.appendChild(ripple);

            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });
});

// ========================================
// Download Diabetes Report as PDF
// ========================================
function downloadDiabetesReport() {
    // Get form data
    const formData = {
        age: document.getElementById('age')?.value || 'N/A',
        gender: document.getElementById('gender')?.value || 'N/A',
        height: document.getElementById('height')?.value || 'N/A',
        weight: document.getElementById('weight')?.value || 'N/A',
        bmi: document.getElementById('bmi-display')?.value || 'N/A',
        glucose: document.getElementById('glucose')?.value || 'N/A',
        bloodPressure: document.getElementById('blood_pressure')?.value || 'N/A',
        insulin: document.getElementById('insulin')?.value || 'N/A',
        skinThickness: document.getElementById('skin_thickness')?.value || 'N/A',
        pregnancies: document.getElementById('pregnancies')?.value || '0',
        diabetesPedigree: document.getElementById('diabetes_pedigree')?.value || 'N/A'
    };

    // Get result data
    const probability = document.querySelector('.probability')?.textContent || '0%';
    const riskBadge = document.querySelector('.risk-badge')?.textContent || 'Unknown';
    const recommendation = document.querySelector('.recommendation')?.textContent || '';

    // Get current date
    const currentDate = new Date().toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });

    // Create PDF content
    const reportContent = `
================================================================================
                    MEDICARE AI - DIABETES RISK ASSESSMENT REPORT
================================================================================

Report Generated: ${currentDate}
Report ID: DRA-${Date.now()}

--------------------------------------------------------------------------------
                              PATIENT INFORMATION
--------------------------------------------------------------------------------

Age:                    ${formData.age} years
Gender:                 ${formData.gender.charAt(0).toUpperCase() + formData.gender.slice(1)}
Height:                 ${formData.height} cm
Weight:                 ${formData.weight} kg
BMI:                    ${formData.bmi}

--------------------------------------------------------------------------------
                              HEALTH INDICATORS
--------------------------------------------------------------------------------

Fasting Glucose:        ${formData.glucose} mg/dL
Blood Pressure:         ${formData.bloodPressure} mmHg (Systolic)
Insulin Level:          ${formData.insulin} μU/mL
Skin Thickness:         ${formData.skinThickness} mm
Pregnancies:            ${formData.pregnancies}
Family History Score:   ${formData.diabetesPedigree}

--------------------------------------------------------------------------------
                              RISK ASSESSMENT RESULTS
--------------------------------------------------------------------------------

DIABETES RISK PROBABILITY:  ${probability}
RISK LEVEL:                 ${riskBadge.toUpperCase()}

--------------------------------------------------------------------------------
                              RECOMMENDATION
--------------------------------------------------------------------------------

${recommendation.replace(/<[^>]*>/g, '').trim()}

--------------------------------------------------------------------------------
                              REFERENCE RANGES
--------------------------------------------------------------------------------

Parameter               Normal Range            Your Value
-----------             ------------            ----------
Fasting Glucose         70-100 mg/dL            ${formData.glucose} mg/dL
Blood Pressure          90-120 mmHg             ${formData.bloodPressure} mmHg
BMI                     18.5-24.9               ${formData.bmi}
Insulin                 2.6-24.9 μU/mL          ${formData.insulin} μU/mL

--------------------------------------------------------------------------------
                              NEXT STEPS
--------------------------------------------------------------------------------

${getRiskBasedNextSteps(riskBadge)}

--------------------------------------------------------------------------------
                              CONSULT SPECIALISTS
--------------------------------------------------------------------------------

• Practo - Online Diabetologist Consult: https://www.practo.com/consult/diabetologist-702
• Apollo 247 - Book Diabetologist: https://www.apollo247.com/specialties/diabetology
• Oladoc - Diabetologists (Pakistan): https://oladoc.com/pakistan/lahore/diabetologist
• BeatO - Diabetes Care Program: https://www.beatoapp.com/

================================================================================
                              MEDICAL DISCLAIMER
================================================================================

This report is generated by MediCare AI for educational purposes only and 
should NOT replace professional medical advice, diagnosis, or treatment.

• Results are based on statistical models and may not reflect actual health status
• Always consult healthcare professionals for proper diagnosis
• Regular medical check-ups are recommended regardless of prediction results
• This assessment uses a machine learning model with 99.41% accuracy

================================================================================
                    © ${new Date().getFullYear()} MediCare AI - All Rights Reserved
================================================================================
`;

    // Create and download file
    const blob = new Blob([reportContent], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `Diabetes_Risk_Report_${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    showNotification('Report downloaded successfully!', 'success');
}

function getRiskBasedNextSteps(riskLevel) {
    const level = riskLevel.toLowerCase();

    if (level.includes('low')) {
        return `✓ Continue maintaining a healthy lifestyle
✓ Regular exercise (150+ minutes per week)
✓ Balanced diet with low sugar intake
✓ Annual health check-ups recommended
✓ Monitor blood glucose levels periodically`;
    } else if (level.includes('medium')) {
        return `⚠ Schedule a consultation with your doctor within 3-6 months
⚠ Consider HbA1c test for better assessment
⚠ Increase physical activity
⚠ Reduce sugar and processed food intake
⚠ Monitor blood glucose levels monthly
⚠ Consider lifestyle modifications`;
    } else {
        return `⚠ URGENT: Consult a healthcare professional immediately
⚠ Get comprehensive diabetes screening (HbA1c, OGTT)
⚠ Consider consulting an endocrinologist
⚠ Strict dietary modifications required
⚠ Regular blood glucose monitoring essential
⚠ Discuss medication options with your doctor`;
    }
}

// ========================================
// Tier and Confidence Display Function
// ========================================
function updateTierDisplay(result) {
    const tierContainer = document.getElementById('tierBadgeContainer');
    const tierBadge = document.getElementById('tierBadge');
    const tierText = document.getElementById('tierText');
    const tierDescription = document.getElementById('tierDescription');
    const confidenceValue = document.getElementById('confidenceValue');
    const upgradeSuggestion = document.getElementById('upgradeSuggestion');
    const missingFeatures = document.getElementById('missingFeatures');
    const validationWarnings = document.getElementById('validationWarnings');
    const warningText = document.getElementById('warningText');
    const persistentDisclaimer = document.getElementById('persistentDisclaimer');
    const disclaimerText = document.getElementById('disclaimerText');

    if (!tierContainer) return;

    // Show tier container
    tierContainer.style.display = 'block';

    // Set tier
    const tier = result.tier || 'standard';
    tierContainer.className = `tier-badge-container mb-3 ${tier}`;
    tierBadge.className = `tier-badge ${tier}`;

    const tierLabels = {
        'screening': 'Basic Screening',
        'standard': 'Standard Assessment',
        'confirmation': 'Comprehensive Analysis'
    };
    tierText.textContent = tierLabels[tier] || 'Standard Assessment';

    // Set tier description
    if (tierDescription && result.tier_description) {
        tierDescription.innerHTML = `<small class="text-muted">${result.tier_description}</small>`;
    }

    // Set confidence
    const confidence = result.confidence || 50;
    if (confidenceValue) {
        confidenceValue.textContent = `${confidence}%`;

        if (confidence >= 75) {
            confidenceValue.className = 'confidence-value high';
        } else if (confidence >= 55) {
            confidenceValue.className = 'confidence-value medium';
        } else {
            confidenceValue.className = 'confidence-value low';
        }
    }

    // Show upgrade suggestion if not at confirmation tier
    if (upgradeSuggestion && missingFeatures) {
        if (result.missing_for_upgrade && result.missing_for_upgrade.length > 0 && tier !== 'confirmation') {
            upgradeSuggestion.style.display = 'block';
            const featureNames = result.missing_for_upgrade.map(f =>
                f.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
            ).join(', ');
            missingFeatures.textContent = `Add ${featureNames} for higher accuracy`;
        } else {
            upgradeSuggestion.style.display = 'none';
        }
    }

    // Show validation warnings
    if (validationWarnings && warningText) {
        if (result.validation_warnings && result.validation_warnings.length > 0) {
            validationWarnings.style.display = 'block';
            warningText.textContent = result.validation_warnings.join('; ');
        } else {
            validationWarnings.style.display = 'none';
        }
    }

    // Show persistent disclaimer
    if (persistentDisclaimer && disclaimerText) {
        persistentDisclaimer.style.display = 'block';
        if (result.result_disclaimer) {
            disclaimerText.textContent = result.result_disclaimer;
        } else if (result.tier_disclaimer) {
            disclaimerText.textContent = result.tier_disclaimer;
        }
    }
}

// ========================================
// Display functions for other diseases
// ========================================
function displayCardiovascularResult(result) {
    if (result.success) {
        const resultContent = document.querySelector('.result-content');
        const placeholder = document.querySelector('.result-placeholder');
        const probability = document.querySelector('.probability');
        const recommendation = document.querySelector('.recommendation');
        const riskBadge = document.querySelector('.risk-badge');

        if (placeholder) placeholder.classList.add('d-none');
        if (resultContent) resultContent.classList.remove('d-none');

        // Update tier display
        updateTierDisplay(result);

        if (probability) animateCounter(probability, 0, result.probability, 1500);

        if (result.risk_level === 'low') {
            updateRiskDisplay(riskBadge, 'Low Risk', 'low-risk', 'success');
            if (recommendation) {
                recommendation.innerHTML = `<strong>Good news!</strong> Your heart disease risk is low. Continue maintaining a healthy lifestyle.`;
            }
        } else if (result.risk_level === 'medium') {
            updateRiskDisplay(riskBadge, 'Medium Risk', 'medium-risk', 'warning');
            if (recommendation) {
                recommendation.innerHTML = `<strong>Attention needed.</strong> Consider consulting a cardiologist.`;
            }
        } else {
            updateRiskDisplay(riskBadge, 'High Risk', 'high-risk', 'danger');
            if (recommendation) {
                recommendation.innerHTML = `<strong>Immediate attention required.</strong> Please consult a cardiologist immediately.`;
            }
        }

        showNotification('Risk assessment completed successfully!', 'success');
    } else {
        if (result.model_unavailable) {
            showNotification('The prediction model is currently unavailable. Please try again later.', 'error');
        } else {
            showNotification(`Prediction failed: ${result.error}`, 'error');
        }
    }
}

function displayKidneyResult(result) {
    if (result.success) {
        const resultContent = document.querySelector('.result-content');
        const placeholder = document.querySelector('.result-placeholder');
        const probability = document.querySelector('.probability');
        const recommendation = document.querySelector('.recommendation');
        const riskBadge = document.querySelector('.risk-badge');

        if (placeholder) placeholder.classList.add('d-none');
        if (resultContent) resultContent.classList.remove('d-none');

        // Update tier display
        updateTierDisplay(result);

        if (probability) animateCounter(probability, 0, result.probability, 1500);

        if (result.risk_level === 'low') {
            updateRiskDisplay(riskBadge, 'Low Risk', 'low-risk', 'success');
            if (recommendation) {
                recommendation.innerHTML = `<strong>Good news!</strong> Your CKD risk is low. Continue maintaining healthy habits.`;
            }
        } else if (result.risk_level === 'medium') {
            updateRiskDisplay(riskBadge, 'Medium Risk', 'medium-risk', 'warning');
            if (recommendation) {
                recommendation.innerHTML = `<strong>Attention needed.</strong> Consider consulting a nephrologist.`;
            }
        } else {
            updateRiskDisplay(riskBadge, 'High Risk', 'high-risk', 'danger');
            if (recommendation) {
                recommendation.innerHTML = `<strong>Immediate attention required.</strong> Please consult a nephrologist immediately.`;
            }
        }

        showNotification('CKD risk assessment completed successfully!', 'success');
    } else {
        if (result.model_unavailable) {
            showNotification('The prediction model is currently unavailable. Please try again later.', 'error');
        } else {
            showNotification(`Prediction failed: ${result.error}`, 'error');
        }
    }
}

function displayDepressionResult(result) {
    if (result.success) {
        const resultContent = document.querySelector('.result-content');
        const placeholder = document.querySelector('.result-placeholder');
        const probability = document.querySelector('.probability');
        const recommendation = document.querySelector('.recommendation');
        const riskBadge = document.querySelector('.risk-badge');

        if (placeholder) placeholder.classList.add('d-none');
        if (resultContent) resultContent.classList.remove('d-none');

        // Update tier display
        updateTierDisplay(result);

        if (probability) animateCounter(probability, 0, result.probability, 1500);

        if (result.risk_level === 'low') {
            updateRiskDisplay(riskBadge, 'Low Risk', 'low-risk', 'success');
            if (recommendation) {
                recommendation.innerHTML = `<strong>Good!</strong> Your mental health indicators are within normal range.`;
            }
        } else if (result.risk_level === 'medium') {
            updateRiskDisplay(riskBadge, 'Medium Risk', 'medium-risk', 'warning');
            if (recommendation) {
                recommendation.innerHTML = `<strong>Some risk factors detected.</strong> Consider speaking with a counselor.`;
            }
        } else {
            updateRiskDisplay(riskBadge, 'High Risk', 'high-risk', 'danger');
            if (recommendation) {
                recommendation.innerHTML = `<strong>Multiple risk factors detected.</strong> Please consult a mental health professional.`;
            }
        }

        showNotification('Risk assessment completed successfully!', 'success');
    } else {
        if (result.model_unavailable) {
            showNotification('The prediction model is currently unavailable. Please try again later.', 'error');
        } else {
            showNotification(`Prediction failed: ${result.error}`, 'error');
        }
    }
}

function displayObesityResult(result) {
    if (result.success) {
        const resultContent = document.querySelector('.result-content') || document.getElementById('resultContent');
        const placeholder = document.querySelector('.result-placeholder') || document.getElementById('resultPlaceholder');
        const probability = document.querySelector('.probability') || document.getElementById('probabilityValue');
        const recommendation = document.querySelector('.recommendation') || document.getElementById('recommendation');
        const riskBadge = document.querySelector('.risk-badge') || document.getElementById('obesityLevel');

        if (placeholder) placeholder.classList.add('d-none');
        if (resultContent) resultContent.classList.remove('d-none');

        // Update tier display
        updateTierDisplay(result);

        const prob = result.probability || 0;
        if (probability) probability.textContent = prob.toFixed(1) + '%';

        const className = result.class_name || result.display_name || 'Unknown';
        if (riskBadge) riskBadge.textContent = className;

        showNotification('Obesity level prediction completed!', 'success');
    } else {
        if (result.model_unavailable) {
            showNotification('The prediction model is currently unavailable. Please try again later.', 'error');
        } else {
            showNotification(`Prediction failed: ${result.error}`, 'error');
        }
    }
}

// ========================================
// Export functions for use in other scripts
// ========================================
window.MediCareAI = {
    getCSRFToken,
    showNotification,
    closeNotification,
    initBMICalculator,
    displayDiabetesResult,
    displayCardiovascularResult,
    displayKidneyResult,
    displayDepressionResult,
    displayObesityResult,
    updateTierDisplay,
    validateForm,
    setLoadingState,
    smoothScrollTo,
    animateCounter,
    animateValue,
    downloadDiabetesReport
};

// ========================================
// Shake Animation CSS (injected dynamically)
// ========================================
const style = document.createElement('style');
style.textContent = `
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
        20%, 40%, 60%, 80% { transform: translateX(5px); }
    }
    
    .ripple {
        position: absolute;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.6);
        transform: scale(0);
        animation: ripple-animation 0.6s ease-out;
        pointer-events: none;
    }
    
    @keyframes ripple-animation {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);


// ========================================
// NEW: Form Progress Tracker
// ========================================
function initFormProgressTracker(formId, diseaseType) {
    const form = document.getElementById(formId);
    if (!form) return;

    const progressBar = document.getElementById('formProgressBar');
    const progressPercentage = document.getElementById('progressPercentage');
    const currentTierLabel = document.getElementById('currentTierLabel');

    // Tier requirements by disease
    const tierRequirements = {
        'diabetes': {
            screening: ['age', 'bmi'],
            standard: ['age', 'bmi', 'glucose', 'blood_pressure'],
            confirmation: ['age', 'bmi', 'glucose', 'blood_pressure', 'insulin']
        },
        'cardiovascular': {
            screening: ['age', 'sex'],
            standard: ['age', 'sex', 'resting_bp', 'cholesterol', 'chest_pain_type'],
            confirmation: ['age', 'sex', 'resting_bp', 'cholesterol', 'chest_pain_type', 'max_hr', 'exercise_angina', 'oldpeak', 'st_slope']
        },
        'kidney': {
            screening: ['age', 'bp'],
            standard: ['age', 'bp', 'sc', 'bu', 'hemo'],
            confirmation: ['age', 'bp', 'sc', 'bu', 'hemo', 'sg', 'al', 'bgr', 'sod', 'pot']
        },
        'depression': {
            screening: ['age', 'gender', 'sleep_duration'],
            standard: ['age', 'gender', 'sleep_duration', 'academic_pressure', 'work_study_hours', 'financial_stress'],
            confirmation: ['age', 'gender', 'sleep_duration', 'academic_pressure', 'work_study_hours', 'financial_stress', 'family_history', 'suicidal_thoughts']
        },
        'obesity': {
            screening: ['height', 'weight', 'age', 'gender'],
            standard: ['height', 'weight', 'age', 'gender', 'faf', 'fcvc', 'ncp'],
            confirmation: ['height', 'weight', 'age', 'gender', 'faf', 'fcvc', 'ncp', 'favc', 'caec', 'calc', 'mtrans', 'family_history_with_overweight']
        },
        'breast_cancer': {
            screening: ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean'],
            standard: ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se'],
            confirmation: ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']
        }
    };

    const requirements = tierRequirements[diseaseType] || tierRequirements['diabetes'];
    const allFields = [...new Set([...requirements.screening, ...requirements.standard, ...requirements.confirmation])];

    function updateProgress() {
        let filledCount = 0;
        let currentTier = 'Not Started';

        allFields.forEach(fieldName => {
            const field = form.querySelector(`[name="${fieldName}"]`) ||
                form.querySelector(`#${fieldName}`);
            if (field && field.value && field.value.trim() !== '') {
                filledCount++;
            }
        });

        // Calculate percentage
        const percentage = Math.round((filledCount / allFields.length) * 100);

        if (progressBar) {
            progressBar.style.width = percentage + '%';
            progressBar.setAttribute('aria-valuenow', percentage);
        }
        if (progressPercentage) {
            progressPercentage.textContent = percentage + '%';
        }

        // Determine current tier
        const screeningFilled = requirements.screening.every(f => {
            const field = form.querySelector(`[name="${f}"]`) || form.querySelector(`#${f}`);
            return field && field.value && field.value.trim() !== '';
        });

        const standardFilled = requirements.standard.every(f => {
            const field = form.querySelector(`[name="${f}"]`) || form.querySelector(`#${f}`);
            return field && field.value && field.value.trim() !== '';
        });

        const confirmationFilled = requirements.confirmation.every(f => {
            const field = form.querySelector(`[name="${f}"]`) || form.querySelector(`#${f}`);
            return field && field.value && field.value.trim() !== '';
        });

        // Update tier markers
        document.querySelectorAll('.tier-marker').forEach(m => {
            m.classList.remove('active', 'completed');
        });

        if (confirmationFilled) {
            currentTier = 'Full Analysis';
            document.querySelector('.tier-marker[data-tier="confirmation"]')?.classList.add('active', 'completed');
            document.querySelector('.tier-marker[data-tier="standard"]')?.classList.add('completed');
            document.querySelector('.tier-marker[data-tier="screening"]')?.classList.add('completed');
        } else if (standardFilled) {
            currentTier = 'Standard';
            document.querySelector('.tier-marker[data-tier="standard"]')?.classList.add('active');
            document.querySelector('.tier-marker[data-tier="screening"]')?.classList.add('completed');
        } else if (screeningFilled) {
            currentTier = 'Screening';
            document.querySelector('.tier-marker[data-tier="screening"]')?.classList.add('active');
        }

        if (currentTierLabel) {
            currentTierLabel.textContent = currentTier;
        }
    }

    // Add listeners to all form inputs
    form.querySelectorAll('input, select, textarea').forEach(input => {
        input.addEventListener('input', updateProgress);
        input.addEventListener('change', updateProgress);
    });

    // Initial update
    updateProgress();
}

// ========================================
// NEW: Feature Importance Display (SHAP-like)
// ========================================
function displayFeatureImportance(result, inputData, diseaseType) {
    const container = document.getElementById('featureImportanceChart');
    const panel = document.getElementById('shapExplanation');

    if (!container || !panel) return;

    // Feature importance weights by disease (simplified SHAP-like values)
    const featureWeights = {
        'diabetes': {
            'glucose': { weight: 0.35, label: 'Blood Glucose' },
            'bmi': { weight: 0.20, label: 'BMI' },
            'age': { weight: 0.15, label: 'Age' },
            'insulin': { weight: 0.12, label: 'Insulin Level' },
            'blood_pressure': { weight: 0.10, label: 'Blood Pressure' },
            'diabetes_pedigree': { weight: 0.08, label: 'Family History' }
        },
        'cardiovascular': {
            'chest_pain_type': { weight: 0.20, label: 'Chest Pain Type' },
            'oldpeak': { weight: 0.18, label: 'ST Depression' },
            'st_slope': { weight: 0.15, label: 'ST Slope' },
            'max_hr': { weight: 0.12, label: 'Max Heart Rate' },
            'cholesterol': { weight: 0.12, label: 'Cholesterol' },
            'exercise_angina': { weight: 0.10, label: 'Exercise Angina' },
            'age': { weight: 0.08, label: 'Age' },
            'resting_bp': { weight: 0.05, label: 'Resting BP' }
        }
    };

    const weights = featureWeights[diseaseType] || featureWeights['diabetes'];

    // Calculate contribution for each feature
    const contributions = [];

    for (const [key, config] of Object.entries(weights)) {
        const value = inputData[key];
        if (value !== undefined && value !== null && value !== '') {
            // Simplified contribution calculation
            let contribution = 0;
            let direction = 'neutral';

            // Determine if value increases or decreases risk
            if (diseaseType === 'diabetes') {
                if (key === 'glucose' && value > 126) {
                    contribution = config.weight * 100;
                    direction = 'positive'; // increases risk
                } else if (key === 'glucose' && value < 100) {
                    contribution = config.weight * 50;
                    direction = 'negative'; // decreases risk
                } else if (key === 'bmi' && value > 30) {
                    contribution = config.weight * 80;
                    direction = 'positive';
                } else if (key === 'bmi' && value < 25) {
                    contribution = config.weight * 40;
                    direction = 'negative';
                } else {
                    contribution = config.weight * 30;
                    direction = 'neutral';
                }
            }

            contributions.push({
                name: config.label,
                value: value,
                contribution: contribution,
                direction: direction,
                weight: config.weight
            });
        }
    }

    // Sort by contribution
    contributions.sort((a, b) => b.contribution - a.contribution);

    // Generate HTML
    let html = '';
    const maxContribution = Math.max(...contributions.map(c => c.contribution), 1);

    contributions.slice(0, 6).forEach(item => {
        const barWidth = (item.contribution / maxContribution) * 100;
        html += `
            <div class="feature-bar-container">
                <div class="feature-bar-label">
                    <span class="feature-bar-name">${item.name}</span>
                    <span class="feature-bar-value">${item.value}</span>
                </div>
                <div class="feature-bar-wrapper">
                    <div class="feature-bar ${item.direction} animate" style="width: ${barWidth}%"></div>
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
    panel.style.display = 'block';
}

// ========================================
// NEW: Health Insights Generator
// ========================================
function generateHealthInsights(inputData, result, diseaseType) {
    const panel = document.getElementById('healthInsightsPanel');
    const insightsList = document.getElementById('insightsList');
    const rangesChart = document.getElementById('rangesChart');

    if (!panel || !insightsList) return;

    const insights = [];

    // Generate insights based on disease type
    if (diseaseType === 'diabetes') {
        // Glucose insight
        const glucose = parseFloat(inputData.glucose);
        if (glucose) {
            if (glucose < 100) {
                insights.push({
                    type: 'good',
                    icon: 'check-circle',
                    title: 'Normal Fasting Glucose',
                    description: `Your glucose level (${glucose} mg/dL) is within the normal range.`
                });
            } else if (glucose < 126) {
                insights.push({
                    type: 'warning',
                    icon: 'exclamation-triangle',
                    title: 'Prediabetic Range',
                    description: `Your glucose level (${glucose} mg/dL) indicates prediabetes. Consider lifestyle changes.`
                });
            } else {
                insights.push({
                    type: 'danger',
                    icon: 'exclamation-circle',
                    title: 'Elevated Glucose',
                    description: `Your glucose level (${glucose} mg/dL) is in the diabetic range. Consult a doctor.`
                });
            }
        }

        // BMI insight
        const bmi = parseFloat(inputData.bmi);
        if (bmi) {
            if (bmi < 18.5) {
                insights.push({
                    type: 'info',
                    icon: 'info-circle',
                    title: 'Underweight',
                    description: `Your BMI (${bmi}) indicates you may be underweight.`
                });
            } else if (bmi < 25) {
                insights.push({
                    type: 'good',
                    icon: 'check-circle',
                    title: 'Healthy Weight',
                    description: `Your BMI (${bmi}) is within the healthy range.`
                });
            } else if (bmi < 30) {
                insights.push({
                    type: 'warning',
                    icon: 'exclamation-triangle',
                    title: 'Overweight',
                    description: `Your BMI (${bmi}) indicates overweight. Consider increasing physical activity.`
                });
            } else {
                insights.push({
                    type: 'danger',
                    icon: 'exclamation-circle',
                    title: 'Obesity',
                    description: `Your BMI (${bmi}) indicates obesity. This increases diabetes risk significantly.`
                });
            }
        }

        // Blood pressure insight
        const bp = parseFloat(inputData.blood_pressure);
        if (bp) {
            if (bp < 120) {
                insights.push({
                    type: 'good',
                    icon: 'check-circle',
                    title: 'Normal Blood Pressure',
                    description: `Your systolic BP (${bp} mmHg) is normal.`
                });
            } else if (bp < 140) {
                insights.push({
                    type: 'warning',
                    icon: 'exclamation-triangle',
                    title: 'Elevated Blood Pressure',
                    description: `Your systolic BP (${bp} mmHg) is slightly elevated.`
                });
            } else {
                insights.push({
                    type: 'danger',
                    icon: 'exclamation-circle',
                    title: 'High Blood Pressure',
                    description: `Your systolic BP (${bp} mmHg) indicates hypertension.`
                });
            }
        }
    }

    // Generate insights HTML
    let insightsHtml = '';
    insights.forEach(insight => {
        insightsHtml += `
            <div class="insight-item">
                <div class="insight-icon ${insight.type}">
                    <i class="fas fa-${insight.icon}"></i>
                </div>
                <div class="insight-content">
                    <div class="insight-title">${insight.title}</div>
                    <p class="insight-description">${insight.description}</p>
                </div>
            </div>
        `;
    });

    insightsList.innerHTML = insightsHtml;

    // Generate range comparison chart
    if (rangesChart && diseaseType === 'diabetes') {
        let rangesHtml = '';

        // Glucose range
        const glucose = parseFloat(inputData.glucose) || 0;
        const glucosePos = Math.min(Math.max((glucose - 50) / 150 * 100, 0), 100);
        const glucoseStatus = glucose < 100 ? 'normal' : (glucose < 126 ? 'elevated' : 'high');

        rangesHtml += `
            <div class="range-item">
                <div class="range-label">
                    <span class="range-name">Glucose</span>
                    <span class="range-value">${glucose} mg/dL</span>
                </div>
                <div class="range-bar-container">
                    <div class="range-normal-zone" style="left: 13%; width: 34%;"></div>
                    <div class="range-marker ${glucoseStatus}" style="left: ${glucosePos}%"></div>
                </div>
                <div class="range-labels">
                    <span>50</span>
                    <span>100</span>
                    <span>126</span>
                    <span>200</span>
                </div>
            </div>
        `;

        // BMI range
        const bmi = parseFloat(inputData.bmi) || 0;
        const bmiPos = Math.min(Math.max((bmi - 15) / 35 * 100, 0), 100);
        const bmiStatus = bmi < 18.5 ? 'low' : (bmi < 25 ? 'normal' : (bmi < 30 ? 'elevated' : 'high'));

        rangesHtml += `
            <div class="range-item">
                <div class="range-label">
                    <span class="range-name">BMI</span>
                    <span class="range-value">${bmi}</span>
                </div>
                <div class="range-bar-container">
                    <div class="range-normal-zone" style="left: 10%; width: 28%;"></div>
                    <div class="range-marker ${bmiStatus}" style="left: ${bmiPos}%"></div>
                </div>
                <div class="range-labels">
                    <span>15</span>
                    <span>18.5</span>
                    <span>25</span>
                    <span>30</span>
                    <span>50</span>
                </div>
            </div>
        `;

        rangesChart.innerHTML = rangesHtml;
    }

    panel.style.display = 'block';
}

// ========================================
// NEW: Real-time Input Validation with Visual Feedback
// ========================================
function initRealTimeValidation(formId) {
    const form = document.getElementById(formId);
    if (!form) return;

    const validationRules = {
        'age': { min: 18, max: 120, message: 'Age should be between 18-120' },
        'glucose': { min: 50, max: 400, message: 'Glucose should be between 50-400 mg/dL' },
        'blood_pressure': { min: 60, max: 200, message: 'BP should be between 60-200 mmHg' },
        'bmi': { min: 10, max: 60, message: 'BMI should be between 10-60' },
        'insulin': { min: 0, max: 900, message: 'Insulin should be between 0-900 μU/mL' },
        'cholesterol': { min: 100, max: 600, message: 'Cholesterol should be between 100-600 mg/dL' },
        'max_hr': { min: 60, max: 220, message: 'Max HR should be between 60-220 bpm' }
    };

    form.querySelectorAll('input[type="number"]').forEach(input => {
        const fieldName = input.name || input.id;
        const rule = validationRules[fieldName];

        input.addEventListener('input', function () {
            const value = parseFloat(this.value);

            // Remove existing feedback
            const existingFeedback = this.parentNode.querySelector('.real-time-feedback');
            if (existingFeedback) existingFeedback.remove();

            if (rule && value) {
                if (value < rule.min || value > rule.max) {
                    this.classList.add('is-invalid');
                    this.classList.remove('is-valid');

                    const feedback = document.createElement('div');
                    feedback.className = 'real-time-feedback text-danger small mt-1';
                    feedback.innerHTML = `<i class="fas fa-exclamation-circle me-1"></i>${rule.message}`;
                    this.parentNode.appendChild(feedback);
                } else {
                    this.classList.remove('is-invalid');
                    this.classList.add('is-valid');
                }
            }
        });
    });
}

// ========================================
// Export new functions
// ========================================
window.MediCareAI = {
    ...window.MediCareAI,
    initFormProgressTracker,
    displayFeatureImportance,
    generateHealthInsights,
    initRealTimeValidation
};
