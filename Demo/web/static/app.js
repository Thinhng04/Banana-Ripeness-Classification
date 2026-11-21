const $ = (id) => document.getElementById(id);
const labels = [
  { name: 'Green', display: 'Green', color: '#68bf35' },
  { name: 'Semi-ripe', display: 'Semi-Ripe', color: '#bfd938' },
  { name: 'Ripe', display: 'Ripe', color: '#ffc900' },
  { name: 'Overripe', display: 'Overripe', color: '#95602e' },
];
const families = ['SVM', 'CNN', 'MobileNetV2'];
let models = [], selected = null, currentFamily = 'CNN';
let photo = null, revision = 0, busy = false;

function message(text = '') { $('status').textContent = text; }

function updateButton() {
  $('predict').disabled = busy || !photo || !selected?.available;
  $('clear').disabled = !photo;
}

function resetResults() {
  $('result-content').hidden = true;
  $('result-empty').hidden = false;
  $('result-empty').querySelector('strong').textContent = photo
    ? 'Ảnh đã sẵn sàng để phân tích'
    : 'Kết quả dự đoán sẽ hiển thị tại đây';
  $('result-empty').querySelector('span:last-child').textContent = photo
    ? 'Chọn mô hình rồi nhấn “Phân tích ảnh”.'
    : 'Tải ảnh lên, chọn mô hình và nhấn “Phân tích ảnh”.';
}

for (const item of labels) {
  const row = document.createElement('div');
  row.className = 'result-row';
  row.innerHTML = `
    <span class="result-label" style="--bar:${item.color}">${item.display}</span>
    <div class="track" role="progressbar" aria-label="${item.name}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0">
      <div class="bar" style="--bar:${item.color}"></div>
    </div>
    <span class="result-value">—</span>`;
  row.dataset.label = item.name;
  $('results').append(row);
}

function renderModelOptions(preferredId) {
  const select = $('model-select');
  const inFamily = models.filter(model => model.family === currentFamily);
  select.replaceChildren();
  for (const model of inFamily) {
    const option = document.createElement('option');
    option.value = model.id;
    option.disabled = !model.available;
    option.textContent = `${model.family} · ${model.name.replace(`${model.family} · `, '')} — ${model.detail}${model.available ? '' : ' (chưa có checkpoint)'}`;
    select.append(option);
  }

  selected = inFamily.find(model => model.id === preferredId && model.available)
    || inFamily.find(model => model.available)
    || inFamily[0]
    || null;
  if (selected) {
    select.value = selected.id;
  }
  updateButton();
}

function setFamily(family) {
  if (!families.includes(family)) return;
  currentFamily = family;
  document.querySelectorAll('.family-tab').forEach(tab => {
    const active = tab.dataset.family === family;
    tab.classList.toggle('active', active);
    tab.setAttribute('aria-selected', String(active));
  });
  renderModelOptions();
  resetResults();
  message();
}

async function loadModels() {
  try {
    const response = await fetch('/api/models');
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Không tải được danh sách mô hình.');
    models = data.models || [];
    const defaultModel = models.find(model => model.id === 'cnn_b' && model.available)
      || models.find(model => model.family === 'CNN' && model.available)
      || models.find(model => model.available);
    if (defaultModel) currentFamily = defaultModel.family;
    document.querySelectorAll('.family-tab').forEach(tab => {
      const active = tab.dataset.family === currentFamily;
      tab.classList.toggle('active', active);
      tab.setAttribute('aria-selected', String(active));
    });
    renderModelOptions(defaultModel?.id);
  } catch (error) {
    message(error.message);
  }
}

document.querySelectorAll('.family-tab').forEach(tab => {
  tab.addEventListener('click', () => setFamily(tab.dataset.family));
});

$('model-select').addEventListener('change', () => {
  selected = models.find(model => model.id === $('model-select').value) || null;
  resetResults();
  message();
  updateButton();
});

async function imageDimensions(file) {
  try {
    const bitmap = await createImageBitmap(file);
    const dimensions = `${bitmap.width}×${bitmap.height}`;
    bitmap.close();
    return dimensions;
  } catch {
    return 'kích thước gốc';
  }
}

async function setPhoto(file) {
  const token = ++revision;
  busy = false;
  photo = null;
  $('predict').textContent = 'Phân tích ảnh';
  $('preview').hidden = true;
  $('placeholder').hidden = false;
  $('image-meta').hidden = true;
  $('dropzone').classList.remove('has-image', 'drag');
  resetResults();
  updateButton();
  message();
  if (!file) return;
  if (file.size > 15 * 1024 * 1024) {
    message('Vui lòng chọn ảnh nhỏ hơn 15 MB.');
    return;
  }

  message('Đang tải ảnh…');
  try {
    const [response, dimensions] = await Promise.all([
      fetch('/api/preview', { method: 'POST', body: file }),
      imageDimensions(file),
    ]);
    const data = await response.json();
    if (token !== revision) return;
    if (!response.ok) throw new Error(data.error || 'Không thể đọc ảnh.');
    photo = file;
    $('preview').src = data.preview;
    $('preview').hidden = false;
    $('placeholder').hidden = true;
    $('image-meta').hidden = false;
    $('image-name').textContent = `${file.name || 'Ảnh chụp'} · ${dimensions}`;
    $('dropzone').classList.add('has-image');
    resetResults();
    updateButton();
    message();
  } catch (error) {
    if (token === revision) message(error.message);
  }
}

$('file').addEventListener('change', event => {
  setPhoto(event.target.files[0]);
  event.target.value = '';
});
$('clear').addEventListener('click', () => setPhoto(null));
$('upload-main').addEventListener('click', event => {
  event.stopPropagation();
  $('file').click();
});
$('dropzone').addEventListener('click', event => {
  if (!event.target.closest('#clear, #upload-main')) $('file').click();
});

for (const eventName of ['dragover', 'dragenter']) {
  $('dropzone').addEventListener(eventName, event => {
    event.preventDefault();
    $('dropzone').classList.add('drag');
  });
}
for (const eventName of ['dragleave', 'drop']) {
  $('dropzone').addEventListener(eventName, event => {
    event.preventDefault();
    $('dropzone').classList.remove('drag');
  });
}
$('dropzone').addEventListener('drop', event => setPhoto(event.dataTransfer.files[0]));

async function run() {
  if (!photo || !selected?.available || busy) return;
  const token = ++revision;
  const model = selected;
  busy = true;
  updateButton();
  resetResults();
  message(`Đang chạy ${model.name}…`);
  $('predict').textContent = 'Đang phân tích…';

  try {
    const response = await fetch(`/api/predict?model=${encodeURIComponent(model.id)}`, {
      method: 'POST',
      body: photo,
    });
    const data = await response.json();
    if (token !== revision) return;
    if (!response.ok) throw new Error(data.error || 'Phân tích ảnh thất bại.');

    for (const result of data.probabilities) {
      const row = $('results').querySelector(`[data-label="${CSS.escape(result.label)}"]`);
      if (!row) continue;
      const percent = result.probability * 100;
      row.querySelector('.result-value').textContent = `${percent.toFixed(1)}%`;
      row.querySelector('.bar').style.width = `${percent}%`;
      row.querySelector('[role="progressbar"]').setAttribute('aria-valuenow', percent.toFixed(1));
    }

    const best = data.probabilities.reduce((a, b) => a.probability > b.probability ? a : b);
    $('verdict').textContent = labels.find(item => item.name === best.label)?.display || best.label;
    const bestColor = labels.find(item => item.name === best.label)?.color || '#68bf35';
    $('prediction-swatch').style.background = bestColor;
    $('confidence-label').textContent = data.has_probabilities ? 'ĐỘ TIN CẬY' : 'NHÃN DỰ ĐOÁN';
    $('confidence').textContent = data.has_probabilities
      ? `${(best.probability * 100).toFixed(1)}%`
      : 'SVM';
    $('model-used').textContent = data.model || model.name;
    $('elapsed').textContent = `Suy luận: ${Math.round(data.elapsed_ms)} ms`;
    $('svm-note').hidden = Boolean(data.has_probabilities);
    $('result-empty').hidden = true;
    $('result-content').hidden = false;
    message();
  } catch (error) {
    if (token === revision) message(error.message);
  } finally {
    if (token === revision) {
      busy = false;
      $('predict').textContent = 'Phân tích ảnh';
      updateButton();
    }
  }
}
$('predict').addEventListener('click', run);
resetResults();
loadModels();
