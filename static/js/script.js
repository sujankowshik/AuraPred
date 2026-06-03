/* 
==================================================================
   ML House Price Prediction Dashboard - Client Controller
   Implements Theme Toggles, Tabs, AJAX, Counter Animations & Cache
==================================================================
*/

document.addEventListener('DOMContentLoaded', () => {
  // --- DOM Elements ---
  const themeToggleBtn = document.getElementById('theme-toggle');
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');
  
  const predictForm = document.getElementById('predict-form');
  const submitBtn = document.getElementById('btn-submit');
  
  // Slider Elements
  const areaSlider = document.getElementById('Area_SqFt');
  const areaVal = document.getElementById('area-val');
  const yearSlider = document.getElementById('Year_Built');
  const yearVal = document.getElementById('year-val');
  
  // Result Panel Views
  const placeholderView = document.getElementById('placeholder-view');
  const loaderView = document.getElementById('loader-view');
  const predictionView = document.getElementById('prediction-view');
  
  // Prediction Outputs
  const predictedPrice = document.getElementById('predicted-price');
  const outModel = document.getElementById('out-model');
  const outAccuracy = document.getElementById('out-accuracy');
  const outArea = document.getElementById('out-area');
  const outBeds = document.getElementById('out-beds');
  const outBaths = document.getElementById('out-baths');
  const outLocation = document.getElementById('out-location');
  const outYear = document.getElementById('out-year');
  const outParking = document.getElementById('out-parking');
  const outFloors = document.getElementById('out-floors');
  const outAmenities = document.getElementById('out-amenities');
  
  // History Elements
  const historyList = document.getElementById('history-list');
  const clearHistoryBtn = document.getElementById('clear-history');

  // --- Theme System ---
  const initTheme = () => {
    const savedTheme = localStorage.getItem('theme') || 'dark'; // Dark theme default for premium feel
    document.documentElement.setAttribute('data-theme', savedTheme);
  };
  
  themeToggleBtn.addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
  });

  initTheme();

  // --- Dynamic Slider Value Indicators ---
  if (areaSlider && areaVal) {
    areaSlider.addEventListener('input', (e) => {
      areaVal.textContent = `${Number(e.target.value).toLocaleString()} sq ft`;
    });
  }
  
  if (yearSlider && yearVal) {
    yearSlider.addEventListener('input', (e) => {
      yearVal.textContent = e.target.value;
    });
  }

  // --- Tabbed Navigation Controller ---
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');
      
      // Update active button state
      tabButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      
      // Update active tab content visibility
      tabContents.forEach(content => {
        content.classList.remove('active');
        if (content.id === `${targetTab}-tab`) {
          content.classList.add('active');
        }
      });
    });
  });

  // --- Dynamic Price Counter Animation ---
  const animatePrice = (targetValue, durationMs = 1200) => {
    const start = 0;
    const end = Math.round(targetValue);
    const range = end - start;
    let current = start;
    const increment = end > start ? Math.ceil(range / (durationMs / 16)) : -1;
    const startTime = performance.now();
    
    const step = (now) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / durationMs, 1);
      
      // Easing out quadratic function
      const easeProgress = progress * (2 - progress);
      current = Math.round(start + easeProgress * range);
      
      predictedPrice.textContent = `₹${current.toLocaleString()}`;
      
      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        predictedPrice.textContent = `₹${end.toLocaleString()}`;
      }
    };
    
    requestAnimationFrame(step);
  };

  // --- Prediction History Pipeline (localStorage Cache) ---
  const getHistory = () => {
    try {
      return JSON.parse(localStorage.getItem('prediction_history')) || [];
    } catch {
      return [];
    }
  };

  const saveToHistory = (item) => {
    const history = getHistory();
    // Prepend new item, keep last 20 queries max
    history.unshift(item);
    if (history.length > 20) history.pop();
    localStorage.setItem('prediction_history', JSON.stringify(history));
    renderHistoryList();
  };

  const renderHistoryList = () => {
    if (!historyList) return;
    const history = getHistory();
    
    if (history.length === 0) {
      historyList.innerHTML = `
        <div class="history-empty">
          <div class="history-empty-icon">📂</div>
          <p>No prediction history yet. Try analyzing a house property!</p>
        </div>
      `;
      if (clearHistoryBtn) clearHistoryBtn.style.display = 'none';
      return;
    }
    
    if (clearHistoryBtn) clearHistoryBtn.style.display = 'block';
    
    historyList.innerHTML = history.map((item, idx) => {
      const dateStr = new Date(item.timestamp).toLocaleString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
      return `
        <div class="history-item" style="animation-delay: ${idx * 0.05}s">
          <div class="history-item-details">
            <div class="history-item-title">${item.area.toLocaleString()} sq ft • ${item.beds} Bed • ${item.baths} Bath</div>
            <div class="history-item-meta">${item.location} • ${item.amenities} Amenities • Built in ${item.year}</div>
            <div class="history-item-meta" style="font-size: 0.7rem; color: var(--text-light); margin-top: 0.2rem;">${dateStr}</div>
          </div>
          <div class="history-item-price">${item.formatted_price}</div>
        </div>
      `;
    }).join('');
  };

  if (clearHistoryBtn) {
    clearHistoryBtn.addEventListener('click', () => {
      if (confirm('Are you sure you want to clear your prediction history?')) {
        localStorage.removeItem('prediction_history');
        renderHistoryList();
      }
    });
  }

  // Render history list on start
  renderHistoryList();

  // --- AJAX Prediction Submission ---
  if (predictForm) {
    predictForm.addEventListener('submit', (e) => {
      e.preventDefault();
      
      // Extract form data
      const formData = new FormData(predictForm);
      const data = {};
      formData.forEach((value, key) => {
        data[key] = value;
      });
      
      // UI Transitions: Show Loader
      placeholderView.style.display = 'none';
      predictionView.style.display = 'none';
      loaderView.style.display = 'flex';
      submitBtn.disabled = true;
      submitBtn.innerHTML = `🔮 Processing Property...`;

      // Simulate a realistic "AI Computing" lag of 1.2 seconds for enhanced UI premium aesthetics
      setTimeout(() => {
        fetch('/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest'
          },
          body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(res => {
          // Reset button
          submitBtn.disabled = false;
          submitBtn.innerHTML = `🔮 Analyze & Predict Price`;
          
          if (res.success) {
            // Hide Loader, Display Result View
            loaderView.style.display = 'none';
            predictionView.style.display = 'flex';
            
            // Animate Counter predicted price
            animatePrice(res.price);
            
            // Populate property specifics
            outModel.textContent = res.model_used;
            outAccuracy.textContent = res.r2_score;
            outArea.textContent = `${res.inputs.Area_SqFt.toLocaleString()} sq ft`;
            outBeds.textContent = `${res.inputs.Bedrooms} Bedrooms`;
            outBaths.textContent = `${res.inputs.Bathrooms} Bathrooms`;
            outLocation.textContent = res.inputs.Location;
            outYear.textContent = res.inputs.Year_Built;
            outParking.textContent = `${res.inputs.Parking} Parking`;
            outFloors.textContent = `${res.inputs.Floors} Floors`;
            outAmenities.textContent = res.inputs.Amenities;
            
            // 1. Update Multi-Model prices comparison grid
            updateMultiModelPrices(res.multi_model_prices);
            
            // 2. Render explainability waterfall progress bars
            renderExplainability(res.explainability_contributions);
            
            // 3. Render Property Matchmaker top recommendations
            renderRecommendedMatches(res.recommended_matches);
            
            // 4. Draw interactive appreciation line chart using Chart.js
            renderTrendChart(res.historical_appreciation_trends);
            
            // 5. Initialize EMI mortgage calculations on predicted value
            emiCalculatedPrice = res.price;
            calcEMI();
            
            // Save to prediction history
            saveToHistory({
              area: res.inputs.Area_SqFt,
              beds: res.inputs.Bedrooms,
              baths: res.inputs.Bathrooms,
              location: res.inputs.Location,
              year: res.inputs.Year_Built,
              amenities: res.inputs.Amenities,
              price: res.price,
              formatted_price: res.formatted_price,
              timestamp: Date.now()
            });

            // Smooth scroll result panel into view on mobile
            if (window.innerWidth <= 1024) {
              const resultCard = document.querySelector('.output-panel');
              if (resultCard) {
                resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
              }
            }
          } else {
            // Display errors if server validations fail
            loaderView.style.display = 'none';
            placeholderView.style.display = 'flex';
            const errorMsg = res.errors ? res.errors.join('\n') : (res.error || 'Server error occurred.');
            alert(`⚠️ Prediction Error:\n\n${errorMsg}`);
          }
        })
        .catch(err => {
          submitBtn.disabled = false;
          submitBtn.innerHTML = `🔮 Analyze & Predict Price`;
          loaderView.style.display = 'none';
          placeholderView.style.display = 'flex';
          alert('⚠️ Network communication failure. Please check your network and try again.');
          console.error(err);
        });
      }, 1300);
    });
  }

  // --- PropTech Advanced Upgrade - Script Implementation ---

  // 1. Initialize Leaflet Map (Centered in New Delhi, India)
  let map, marker;
  try {
    const mapContainer = document.getElementById('map');
    if (mapContainer) {
      map = L.map('map', {
        center: [28.6139, 77.2090],
        zoom: 12,
        zoomControl: false
      });
      
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors'
      }).addTo(map);

      // Add dynamic dragging marker
      marker = L.marker([28.6139, 77.2090], { draggable: true }).addTo(map);

      // Sync pin drag values
      marker.on('dragend', function () {
        const position = marker.getLatLng();
        syncMapToLocation(position.lat, position.lng);
      });

      // Sync map click values
      map.on('click', function (event) {
        const latlng = event.latlng;
        marker.setLatLng(latlng);
        syncMapToLocation(latlng.lat, latlng.lng);
      });

      // 1.1 Add Map Location Search functionality
      const searchInput = document.getElementById('map-search-input');
      const searchBtn = document.getElementById('btn-map-search');
      const searchError = document.getElementById('map-search-error');

      const performMapSearch = async () => {
        if (!searchInput || !map || !marker) return;
        const query = searchInput.value.trim();
        if (!query) return;

        searchError.style.display = 'none';
        searchBtn.disabled = true;
        searchBtn.innerHTML = `<span>⏳ Searching...</span>`;

        try {
          // Use OpenStreetMap Nominatim API for geocoding
          const response = await fetch(
            `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=1`,
            {
              headers: {
                'Accept': 'application/json',
                'User-Agent': 'AuraPred/1.0 (RealEstate Valuation Application)'
              }
            }
          );
          
          if (!response.ok) {
            throw new Error("API network response was not ok");
          }

          const data = await response.json();
          if (data && data.length > 0) {
            const lat = parseFloat(data[0].lat);
            const lon = parseFloat(data[0].lon);
            const displayName = data[0].display_name;

            // Pan and Zoom map to search location
            map.setView([lat, lon], 14);
            marker.setLatLng([lat, lon]);
            syncMapToLocation(lat, lon);

            // Open a beautiful Leaflet popup at the marker with searched address
            marker.bindPopup(`🔍 <strong>Searched:</strong> ${displayName.split(',')[0]}<br>📍 Location mapped automatically.`).openPopup();
          } else {
            searchError.textContent = `⚠️ Location not found. Try entering a city or zip code (e.g. "Delhi", "Mumbai").`;
            searchError.style.display = 'block';
          }
        } catch (err) {
          console.error("Geocoding failed: ", err);
          searchError.textContent = `❌ Unable to search location at this time. Please check your internet connection or try again.`;
          searchError.style.display = 'block';
        } finally {
          searchBtn.disabled = false;
          searchBtn.innerHTML = `🔍 Search`;
        }
      };

      if (searchBtn && searchInput) {
        searchBtn.addEventListener('click', performMapSearch);
        searchInput.addEventListener('keydown', (e) => {
          if (e.key === 'Enter') {
            e.preventDefault();
            performMapSearch();
          }
        });
      }

      // 1.2 Geolocation Detection (Detect Current Location)
      const detectBtn = document.getElementById('btn-map-detect');

      const performGeolocation = () => {
        if (!navigator.geolocation) {
          searchError.textContent = `❌ Geolocation is not supported by your browser.`;
          searchError.style.display = 'block';
          return;
        }

        searchError.style.display = 'none';
        detectBtn.disabled = true;
        detectBtn.innerHTML = `<span>⏳ Locating...</span>`;

        navigator.geolocation.getCurrentPosition(
          (position) => {
            const lat = position.coords.latitude;
            const lon = position.coords.longitude;

            // Pan and Zoom map to user location
            map.setView([lat, lon], 14);
            marker.setLatLng([lat, lon]);
            syncMapToLocation(lat, lon);

            // Open popup stating dynamic coordinates
            marker.bindPopup(`📍 <strong>Detected Current Location</strong><br>Coordinates: ${lat.toFixed(4)}, ${lon.toFixed(4)}`).openPopup();

            detectBtn.disabled = false;
            detectBtn.innerHTML = `📍 Locate Me`;
          },
          (err) => {
            console.warn("Geolocation failed: ", err);
            let errMsg = `⚠️ Unable to retrieve your location.`;
            if (err.code === 1) {
              errMsg = `🔒 Location access denied. Please enable location permissions in your browser.`;
            } else if (err.code === 2) {
              errMsg = `❌ Location unavailable. Check your device's GPS or network connection.`;
            } else if (err.code === 3) {
              errMsg = `⏳ Geolocation request timed out. Please try again.`;
            }
            searchError.textContent = errMsg;
            searchError.style.display = 'block';
            
            detectBtn.disabled = false;
            detectBtn.innerHTML = `📍 Locate Me`;
          },
          {
            enableHighAccuracy: true,
            timeout: 10000,
            maximumAge: 0
          }
        );
      };

      if (detectBtn) {
        detectBtn.addEventListener('click', performGeolocation);
      }
    }
  } catch (err) {
    console.error("Leaflet.js mapping load skipped: ", err);
  }

  function syncMapToLocation(lat, lng) {
    const locSelect = document.getElementById('Location');
    if (!locSelect) return;
    
    const centerLat = 28.6139;
    const centerLng = 77.2090;
    const dLat = lat - centerLat;
    const dLng = lng - centerLng;
    
    // Map sectors to locations
    let sector = 'Suburbs';
    if (Math.abs(dLat) < 0.015 && Math.abs(dLng) < 0.015) {
      sector = 'Downtown';
    } else if (dLat > 0.015 && dLng > 0.015) {
      sector = 'Waterfront';
    } else if (dLat > 0.015 && dLng <= 0.015) {
      sector = 'Uptown';
    } else if (dLat <= -0.015 && dLng > 0.015) {
      sector = 'Suburbs';
    } else {
      sector = 'Rural';
    }
    
    locSelect.value = sector;
    locSelect.dispatchEvent(new Event('change'));
    
    // Flash dynamic notification on map marker
    marker.bindPopup(`📍 Drop Pin: Mapped to <strong>${sector}</strong> sector`).openPopup();
  }

  // 2. Interactive Mortgage EMI calculations
  let emiCalculatedPrice = 42240000; // Default (INR scaled stats median)
  
  const calcEMI = () => {
    const downPayPercent = Number(document.getElementById('emi-downpayment').value);
    const interestRateVal = Number(document.getElementById('emi-rate').value);
    const tenureYears = Number(document.getElementById('emi-tenure').value);
    
    const emiDownpaymentVal = document.getElementById('emi-downpayment-val');
    const emiRateVal = document.getElementById('emi-rate-val');
    const emiTenureVal = document.getElementById('emi-tenure-val');
    const emiMonthly = document.getElementById('emi-monthly-val');
    const emiDownpayNum = document.getElementById('emi-downpay-num');
    const emiSalary = document.getElementById('emi-salary-val');
    
    if (!emiDownpaymentVal) return; // Guard
    
    emiDownpaymentVal.textContent = `${downPayPercent}%`;
    emiRateVal.textContent = `${interestRateVal}%`;
    emiTenureVal.textContent = `${tenureYears} Yrs`;
    
    const principal = emiCalculatedPrice * (1 - downPayPercent / 100);
    emiDownpayNum.textContent = `₹${Math.round(emiCalculatedPrice * (downPayPercent / 100)).toLocaleString()}`;
    
    const monthlyRate = interestRateVal / 12 / 100;
    const totalMonths = tenureYears * 12;
    
    let emi = 0;
    if (monthlyRate > 0) {
      emi = principal * monthlyRate * Math.pow(1 + monthlyRate, totalMonths) / (Math.pow(1 + monthlyRate, totalMonths) - 1);
    } else {
      emi = principal / totalMonths;
    }
    
    emiMonthly.textContent = `₹${Math.round(emi).toLocaleString()}`;
    // Safe recommend salary requirement: Salary = EMI * 2.85 (EMI should be at most 35% of monthly salary)
    emiSalary.textContent = `₹${Math.round(emi * 2.85).toLocaleString()}`;
  };
  
  // Hook EMI range inputs
  ['emi-downpayment', 'emi-rate', 'emi-tenure'].forEach(id => {
    const slider = document.getElementById(id);
    if (slider) slider.addEventListener('input', calcEMI);
  });

  // Calculate once on startup
  calcEMI();

  // 3. Render Explainability progress bars (Waterfall influence)
  const renderExplainability = (contributions) => {
    const container = document.getElementById('waterfall-container');
    if (!container) return;
    if (!contributions || contributions.length === 0) {
      container.innerHTML = `<p style="font-size: 0.85rem; color: var(--text-muted); text-align: left;">Explainability contributions analysis not available.</p>`;
      return;
    }
    
    // Scale weights relative to max absolute contribution of ₹1.5 Crores
    const maxVal = 15000000;
    
    container.innerHTML = contributions.map((item, idx) => {
      const isPos = item.weight >= 0;
      const absWeight = Math.abs(item.weight);
      const percent = Math.min(100, (absWeight / maxVal) * 100);
      const sign = isPos ? '+' : '-';
      const colorClass = isPos ? 'contrib-pos' : 'contrib-neg';
      
      return `
        <div class="waterfall-row" style="animation: slideIn 0.3s ease-out forwards; animation-delay: ${idx * 0.05}s">
          <div class="waterfall-label-row">
            <span>${item.feature}</span>
            <span style="color: var(--${isPos ? 'success' : 'danger'})">${sign}₹${Math.round(absWeight).toLocaleString()}</span>
          </div>
          <div class="waterfall-bar-bg">
            <div class="waterfall-bar-fill ${colorClass}" style="width: ${percent}%"></div>
          </div>
        </div>
      `;
    }).join('');
  };

  // 4. Update Multi-Model Price selector grids
  const updateMultiModelPrices = (prices) => {
    const gbVal = document.getElementById('multimodel-gb-val');
    const lrVal = document.getElementById('multimodel-lr-val');
    const rfVal = document.getElementById('multimodel-rf-val');
    
    if (gbVal && prices['Gradient Boosting']) gbVal.textContent = prices['Gradient Boosting'];
    if (lrVal && prices['Linear Regression']) lrVal.textContent = prices['Linear Regression'];
    if (rfVal && prices['Random Forest']) rfVal.textContent = prices['Random Forest'];
  };

  // 5. Appreciation trends lines rendering using Chart.js
  let trendChart = null;
  const renderTrendChart = (trends) => {
    const canvas = document.getElementById('trend-chart-canvas');
    if (!canvas) return;
    
    const years = trends.map(t => t.year);
    const prices = trends.map(t => t.price);
    
    try {
      if (trendChart) trendChart.destroy();
      
      const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
      const textColor = isDark ? '#94a3b8' : '#64748b';
      const gridColor = isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.05)';
      
      trendChart = new Chart(canvas, {
        type: 'line',
        data: {
          labels: years,
          datasets: [{
            label: 'Property Value Growth',
            data: prices,
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            borderWidth: 3,
            fill: true,
            tension: 0.35,
            pointRadius: 4,
            pointBackgroundColor: '#8b5cf6'
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: function(context) {
                  return `Estimated Value: ₹${context.raw.toLocaleString()}`;
                }
              }
            }
          },
          scales: {
            x: {
              grid: { color: gridColor },
              ticks: { color: textColor, font: { family: 'Inter' } }
            },
            y: {
              grid: { color: gridColor },
              ticks: {
                color: textColor,
                font: { family: 'Inter' },
                callback: function(value) {
                  if (value >= 10000000) return `₹${(value/10000000).toFixed(1)}Cr`;
                  if (value >= 100000) return `₹${(value/100000).toFixed(0)}L`;
                  return `₹${value.toLocaleString()}`;
                }
              }
            }
          }
        }
      });
    } catch (err) {
      console.error("Chart.js line chart render skipped: ", err);
    }
  };

  // 6. Property Matchmaker recommended cards lists
  const renderRecommendedMatches = (matches) => {
    const container = document.getElementById('matchmaker-grid');
    if (!container) return;
    if (!matches || matches.length === 0) {
      container.innerHTML = `<p style="grid-column: span 3; color: var(--text-muted); text-align: center; padding: 2rem 0;">No matching recommended listings found.</p>`;
      return;
    }
    
    container.innerHTML = matches.map((item, idx) => {
      return `
        <div class="matchmaker-card" style="animation: slideIn 0.3s ease-out forwards; animation-delay: ${idx * 0.1}s">
          <div class="matchmaker-header">
            <span class="matchmaker-loc">📍 ${item.location}</span>
            <span class="matchmaker-amen">${item.amenities} grade</span>
          </div>
          <div>
            <div class="matchmaker-price">${item.formatted_price}</div>
            <p style="font-size: 0.75rem; color: var(--text-light); margin-top: 0.25rem;">Market Listing Price</p>
          </div>
          <div class="matchmaker-specs">
            <div class="matchmaker-spec-item">📐 ${item.area.toLocaleString()} sqft</div>
            <div class="matchmaker-spec-item">🛏️ ${item.beds} Beds</div>
            <div class="matchmaker-spec-item">🛀 ${item.baths} Baths</div>
            <div class="matchmaker-spec-item">🏗️ Built ${item.year}</div>
            <div class="matchmaker-spec-item">🚗 ${item.parking === 'Yes' ? 'Garage' : 'No'}</div>
            <div class="matchmaker-spec-item">🏢 ${item.floors} Flrs</div>
          </div>
        </div>
      `;
    }).join('');
  };
});
