"""USB capture on S100, latest-frame inference, GTK fullscreen display."""
from pathlib import Path
import os,sys,threading,time,json,re,subprocess,urllib.request
from collections import deque
from resource_monitor import ResourceMonitor,corner_action
from display_settings import DEFAULT_SETTINGS,load_settings,save_settings,cover_geometry,output_dimensions
from festival_settings import load_options,save_options,wall_geometry,list_printers
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
import cv2,numpy as np,gi
cv2.setNumThreads(1)  # Gesture/UI process only; BPU workers configure their own threads.
gi.require_version('Gtk','3.0')
gi.require_version('PangoCairo','1.0')
from gi.repository import Gtk,Gdk,GdkPixbuf,GLib,Pango,PangoCairo

class Player:
 def __init__(self,start_workers=True):
  self.lock=threading.Lock();self.latest=None;self.sequence=0;self.result=None;self.message='等待 USB 摄像头接入…';self.running=True;self.capture_info={};self.painted=None;self.paint_size=None;self.heartbeat=0
  self.settings=load_settings(ROOT/'display-settings.json')
  self.festival_settings=load_options(ROOT/'festival-settings.json')
  self.window=Gtk.Window(title='S100 实时动漫相机');self.window.set_default_size(1280,720)
  self.window.connect('destroy',self.close);self.window.connect('key-press-event',self.key)
  self.window.connect('window-state-event',self.window_state)
  self.frame_times=deque(maxlen=120);self.resources={};self.panel_visible=False;self.cursor_hidden=False;self.frame_status={}
  self.window.add_events(Gdk.EventMask.POINTER_MOTION_MASK)
  self.window.connect('motion-notify-event',self.pointer_motion)
  css=Gtk.CssProvider();css.load_from_data(b'window {background: black;} label {color: white; font-size: 28px;}')
  Gtk.StyleContext.add_provider_for_screen(Gdk.Screen.get_default(),css,Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
  self.box=Gtk.Box(orientation=Gtk.Orientation.VERTICAL);self.image=Gtk.DrawingArea();self.image.set_size_request(1,1);self.image.connect('draw',self.draw_frame);self.pixbuf=None;self.label=Gtk.Label(label=self.message)
  self.festival=None;self.countdown='';self.gallery_paths=[];self.gallery_cache={};self.wall_size=None;self.wall_pending=False
  self.content=Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL,spacing=8);self.content.pack_start(self.image,True,True,0)
  self.sidebar=Gtk.Box(orientation=Gtk.Orientation.VERTICAL,spacing=8);self.sidebar.set_size_request(235,1)
  self.gallery=Gtk.DrawingArea();self.gallery.set_size_request(1,1);self.gallery.connect('draw',self.draw_gallery);self.sidebar.pack_start(self.gallery,True,True,0)
  self.content.connect('size-allocate',self.schedule_wall_layout)
  self.content.pack_end(self.sidebar,False,False,6);self.box.pack_start(self.content,True,True,0);self.box.pack_end(self.label,False,False,12)
  self.overlay=Gtk.Overlay();self.overlay.add(self.box)
  self.panel=Gtk.EventBox();self.panel.set_halign(Gtk.Align.START);self.panel.set_valign(Gtk.Align.START);self.panel.set_margin_start(18);self.panel.set_margin_top(18)
  self.panel.connect('draw',self.draw_panel);self.panel.add_events(Gdk.EventMask.POINTER_MOTION_MASK);self.panel.connect('motion-notify-event',self.pointer_motion)
  self.panel_content=Gtk.Box(orientation=Gtk.Orientation.VERTICAL);self.panel_content.set_name('camera-settings')
  self.panel_scroll=Gtk.ScrolledWindow();self.panel_scroll.set_policy(Gtk.PolicyType.NEVER,Gtk.PolicyType.AUTOMATIC);self.panel_scroll.set_propagate_natural_height(True);self.panel_scroll.set_max_content_height(650);self.panel_scroll.add(self.panel_content);self.panel.add(self.panel_scroll)
  self.window.connect('size-allocate',lambda *_:GLib.idle_add(self.resize_panel))
  self.info=Gtk.DrawingArea();self.info.set_size_request(480,360);self.info.connect('draw',self.draw_info);self.info_text='';self.panel_content.pack_start(self.info,False,False,0)
  self.build_settings()
  self.build_festival_settings()
  panel_css=Gtk.CssProvider();panel_css.load_from_data(b'eventbox {background-color: transparent;border-radius:12px;} #camera-settings label {color:white;font-size:17px;} #camera-settings button,#camera-settings entry,#camera-settings spinbutton {background-image:none;background-color:rgba(25,30,40,0.92);color:white;font-size:17px;}')
  self.panel.get_style_context().add_provider(panel_css,Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION);self.info.get_style_context().add_provider(panel_css,Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
  self.panel_content.get_style_context().add_provider(panel_css,Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
  Gtk.StyleContext.add_provider_for_screen(Gdk.Screen.get_default(),panel_css,Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
  self.panel.show_all();self.panel.hide();self.panel.set_no_show_all(True);self.overlay.add_overlay(self.panel);self.overlay.set_overlay_pass_through(self.panel,False);self.window.add(self.overlay)
  self.window.show_all()
  if start_workers:
   from festival import Festival
   def latest_for_gesture():
    with self.lock:return self.sequence,self.latest
   self.festival=Festival(latest_for_gesture,lambda photo:GLib.idle_add(self.add_photo,photo),lambda:self.running,configuration=lambda:dict(self.festival_settings))
   for photo in self.festival.history:self.add_photo(photo)
   self.window.fullscreen()
   self.capture_thread=threading.Thread(target=self.capture,daemon=True);self.capture_thread.start();self.processing_thread=threading.Thread(target=self.process,daemon=True);self.processing_thread.start();GLib.timeout_add(40,self.paint)
   threading.Thread(target=self.monitor_resources,daemon=True).start();GLib.timeout_add(500,self.refresh_info)
 def pointer_motion(self,window,event):
  _,x,y,_=self.window.get_window().get_device_position(event.get_device())
  state=corner_action(x,y,self.window.get_allocated_width(),self.window.get_allocated_height(),self.panel_visible,max(480,self.panel.get_allocated_width()+36),max(300,self.panel.get_allocated_height()+36))
  self.panel_visible=state['show_panel'];self.panel.set_visible(self.panel_visible)
  if state['hide_cursor']!=self.cursor_hidden:
   self.cursor_hidden=state['hide_cursor'];self.window.get_window().set_cursor(Gdk.Cursor.new_for_display(self.window.get_display(),Gdk.CursorType.BLANK_CURSOR) if self.cursor_hidden else None)
 def close(self,*_):
  self.running=False
  if Gtk.main_level()>0:Gtk.main_quit()
 def add_photo(self,photo):
  photo=Path(photo);self.gallery_paths=([photo]+[p for p in self.gallery_paths if p!=photo])[:100]
  self.schedule_wall_layout();self.gallery.queue_draw()
  return False
 def schedule_wall_layout(self,*_):
  if not self.wall_pending:self.wall_pending=True;GLib.idle_add(self.layout_wall)
 def gallery_geometry(self):
  aspect=16/9
  if self.gallery_paths:
   try:
    info=GdkPixbuf.Pixbuf.get_file_info(str(self.gallery_paths[0]));aspect=info[1]/max(1,info[2])
   except Exception:pass
  return wall_geometry(self.content.get_allocated_width(),self.content.get_allocated_height(),self.festival_settings['gallery_count'],aspect)
 def layout_wall(self):
  self.wall_pending=False;geometry=self.gallery_geometry()
  if self.wall_size!=geometry['width']:self.wall_size=geometry['width'];self.sidebar.set_size_request(self.wall_size,1)
  self.gallery.queue_draw()
  if hasattr(self,'festival_note'):
   self.festival_note.set_text(f"当前最多 {geometry['count']} 张，照片墙宽 {geometry['width']} 像素；新照片在上。")
  return False
 def draw_gallery(self,widget,context):
  geometry=self.gallery_geometry();width=widget.get_allocated_width();height=widget.get_allocated_height()
  context.set_source_rgb(0,0,0);context.paint()
  visible=self.gallery_paths[:geometry['count']]
  for path in list(self.gallery_cache):
   if path not in visible:del self.gallery_cache[path]
  for index,path in enumerate(visible):
   top=index*(geometry['row_height']+geometry['gap']);row=min(geometry['row_height'],height-top)
   if row<=0:break
   try:
    if path not in self.gallery_cache:self.gallery_cache[path]=GdkPixbuf.Pixbuf.new_from_file_at_scale(str(path),900,900,True)
    pixbuf=self.gallery_cache[path];w=pixbuf.get_width();h=pixbuf.get_height();scale=min(width/w,row/h)
    context.save();context.rectangle(0,top,width,row);context.clip();context.translate((width-w*scale)/2,top+(row-h*scale)/2);context.scale(scale,scale)
    Gdk.cairo_set_source_pixbuf(context,pixbuf,0,0);context.paint();context.restore()
   except Exception as error:print('GALLERY',str(error),flush=True)
  return False
 def build_festival_settings(self):
  self.festival_expander=Gtk.Expander(label='照片墙与打印');self.festival_expander.set_margin_start(18);self.festival_expander.set_margin_end(18);self.festival_expander.set_margin_bottom(14)
  grid=Gtk.Grid(column_spacing=10,row_spacing=10);grid.set_margin_top(12);self.festival_expander.add(grid);self.panel_content.pack_start(self.festival_expander,False,False,0)
  self.photo_count=Gtk.SpinButton.new_with_range(0,20,1);self.photo_count.set_numeric(True);self.photo_count.set_value(self.festival_settings['gallery_count'])
  grid.attach(Gtk.Label(label='照片数量',xalign=0),0,0,1,1);grid.attach(self.photo_count,1,0,1,1);grid.attach(Gtk.Label(label='0 = 自动容纳'),2,0,1,1)
  self.pdf_only=Gtk.CheckButton(label='只生成 PDF，不打印');self.pdf_only.set_active(not self.festival_settings['print_enabled']);grid.attach(self.pdf_only,0,1,3,1)
  self.printer_choice=Gtk.ComboBoxText();self.printer_choice.append(self.festival_settings['printer'],self.festival_settings['printer']);self.printer_choice.set_active_id(self.festival_settings['printer'])
  grid.attach(Gtk.Label(label='打印机',xalign=0),0,2,1,1);grid.attach(self.printer_choice,1,2,2,1)
  refresh=Gtk.Button(label='刷新打印机');refresh.connect('clicked',lambda *_:self.refresh_printers());grid.attach(refresh,0,3,1,1)
  apply=Gtk.Button(label='应用并保存');apply.connect('clicked',self.apply_festival_settings);grid.attach(apply,1,3,2,1)
  self.festival_note=Gtk.Label(label='照片墙宽度随数量和屏幕尺寸调整。',xalign=0);self.festival_note.set_line_wrap(True);self.festival_note.set_max_width_chars(35);grid.attach(self.festival_note,0,4,3,1)
  self.print_note=Gtk.Label(label='');self.print_note.set_line_wrap(True);self.print_note.set_max_width_chars(35);self.print_note.set_xalign(0);grid.attach(self.print_note,0,5,3,1)
  self.pdf_only.connect('toggled',lambda button:self.printer_choice.set_sensitive(not button.get_active()));self.printer_choice.set_sensitive(not self.pdf_only.get_active())
  self.festival_expander.connect('notify::expanded',lambda *_:GLib.idle_add(self.resize_panel))
  self.refresh_printers()
 def refresh_printers(self):
  def worker():GLib.idle_add(self.populate_printers,list_printers())
  threading.Thread(target=worker,daemon=True).start()
 def populate_printers(self,names):
  selected=self.printer_choice.get_active_id() or self.festival_settings['printer']
  self.printer_choice.remove_all()
  for name in sorted(set(names+[selected])):self.printer_choice.append(name,name if name in names else name+'（未发现）')
  self.printer_choice.set_active_id(selected);return False
 def apply_festival_settings(self,*_):
  value={'gallery_count':self.photo_count.get_value_as_int(),'print_enabled':not self.pdf_only.get_active(),'printer':self.printer_choice.get_active_id() or self.festival_settings['printer']}
  try:
   self.festival_settings=save_options(ROOT/'festival-settings.json',value);self.schedule_wall_layout()
   self.print_note.set_text('设置已保存；下一张照片仅保存 PDF。' if not value['print_enabled'] else '设置已保存；下一张使用 '+value['printer']+' 打印。')
  except (ValueError,OSError) as error:self.print_note.set_text('设置未保存：'+str(error))
 def build_settings(self):
  self.settings_expander=Gtk.Expander(label='画面设置');self.settings_expander.set_margin_start(18);self.settings_expander.set_margin_end(18);self.settings_expander.set_margin_bottom(14)
  grid=Gtk.Grid(column_spacing=10,row_spacing=10);grid.set_margin_top(12);self.settings_expander.add(grid);self.panel_content.pack_start(self.settings_expander,False,False,0)
  self.presets=[640,960,1280,1920]
  self.preset=Gtk.ComboBoxText();self.preset.append_text('自定义')
  for edge in self.presets:self.preset.append_text(f'长边 {edge} 像素')
  grid.attach(Gtk.Label(label='输出分辨率',xalign=0),0,0,1,1);grid.attach(self.preset,1,0,3,1)
  self.output_width=Gtk.SpinButton.new_with_range(320,1920,4);self.output_width.set_numeric(True);self.output_width.set_digits(0)
  grid.attach(Gtk.Label(label='输出长边',xalign=0),0,1,1,1);grid.attach(self.output_width,1,1,1,1);grid.attach(Gtk.Label(label='宽高自动匹配'),2,1,2,1)
  self.rotation=Gtk.ComboBoxText()
  for angle in (0,90,180,270):self.rotation.append_text(f'{angle}°' + (' 顺时针' if angle else ''))
  grid.attach(Gtk.Label(label='摄像头旋转',xalign=0),0,2,1,1);grid.attach(self.rotation,1,2,3,1)
  self.mirror=Gtk.CheckButton(label='镜像显示');grid.attach(self.mirror,0,3,4,1)
  buttons=Gtk.Box(spacing=10);apply=Gtk.Button(label='应用并保存');apply.connect('clicked',self.apply_settings);reset=Gtk.Button(label='恢复默认');reset.connect('clicked',self.reset_settings);buttons.pack_start(apply,True,True,0);buttons.pack_start(reset,True,True,0);grid.attach(buttons,0,4,4,1)
  self.settings_note=Gtk.Label(label='分辨率只改变像素密度，取景比例保持不变。',xalign=0);self.settings_note.set_line_wrap(True);grid.attach(self.settings_note,0,5,4,1)
  self.preset.connect('changed',self.select_preset);self.sync_settings_controls()
  self.settings_expander.connect('notify::expanded',lambda *_:GLib.idle_add(self.resize_panel))
 def resize_panel(self):
  limit=max(160,self.window.get_allocated_height()-36)
  if self.panel_scroll.get_min_content_height()>limit:self.panel_scroll.set_min_content_height(0)
  self.panel_scroll.set_max_content_height(limit)
  wanted=min(limit,max(40,self.panel_content.get_preferred_height()[1]))
  if self.panel_scroll.get_min_content_height()!=wanted:self.panel_scroll.set_min_content_height(wanted)
  return False
 def select_preset(self,combo):
  index=combo.get_active()-1
  if 0<=index<len(self.presets):
   self.output_width.set_value(self.presets[index])
 def sync_settings_controls(self):
  value=self.settings;edge=max(value['width'],value['height']);self.preset.set_active(self.presets.index(edge)+1 if edge in self.presets else 0)
  self.output_width.set_value(edge);self.rotation.set_active(value['rotation']//90);self.mirror.set_active(value['mirror'])
 def apply_settings(self,*_):
  rotation=self.rotation.get_active()*90;edge=int(round(self.output_width.get_value()/4)*4)
  width=int(self.capture_info.get('width') or 1920);height=int(self.capture_info.get('height') or 1080)
  if rotation in (90,270):width,height=height,width
  width,height=output_dimensions(width,height,edge)
  value={'width':width,'height':height,'rotation':rotation,'mirror':self.mirror.get_active()}
  try:
   self.settings=save_settings(ROOT/'display-settings.json',value);self.sync_settings_controls();self.settings_note.set_text(f'已保存：{width} × {height}，保持原取景比例。')
  except (ValueError,OSError) as error:self.settings_note.set_text('设置未保存：'+str(error))
 def reset_settings(self,*_):
  self.settings=dict(DEFAULT_SETTINGS);self.sync_settings_controls();self.apply_settings()
 def monitor_resources(self):
  monitor=ResourceMonitor();history=deque(maxlen=5)
  while self.running:
   sample=monitor.sample();history.append(sample.get('bpu_percent',{}));keys=sample.get('bpu_percent',{})
   sample['bpu_average_percent']={key:sum(row[key] for row in history if key in row)/sum(key in row for row in history) for key in keys}
   self.resources=sample;time.sleep(1)
 def refresh_info(self):
  r=self.resources;s=self.frame_status;out=s.get('output',[]);fps=s.get('fps',0)
  lines=['实时信息 · S100 BPU',f'输出  {out[0]} × {out[1]}   |   {fps:.1f} FPS' if out else '输出  等待画面',f"采集  {int(self.capture_info.get('width',0))} × {int(self.capture_info.get('height',0))}",f"CPU  {r['cpu_percent']:.1f}%" if 'cpu_percent' in r else 'CPU  采样中']
  lines.insert(2,f'显示  {self.image.get_allocated_width()} × {self.image.get_allocated_height()} · 铺满裁剪')
  lines.append('BPU  '+' / '.join(f"{v:.0f}% · 近5秒均值 {r.get('bpu_average_percent',{}).get(k,v):.0f}%" for k,v in r.get('bpu_percent',{}).items()) if r.get('bpu_percent') else 'BPU  暂无读数')
  if 'bpu_frequency_mhz' in r:lines.append(f"BPU 频率  {r['bpu_frequency_mhz']:.0f} MHz")
  if 'processing_ms' in s:lines.append(f"模型与后处理  {s['processing_ms']:.0f} ms")
  if 'latency_ms' in s:lines.append(f"单帧处理总耗时  {s['latency_ms']:.0f} ms")
  if 'faces' in s:lines.append(f"人脸增强  {s['faces']} 人")
  if 'memory_used_gb' in r:lines.append(f"内存  {r['memory_used_gb']:.2f} / {r['memory_total_gb']:.2f} GB")
  temps=r.get('temperatures',{});soc=[v for k,v in temps.items() if 'cmn' in k];bpu=[v for k,v in temps.items() if 'bpu' in k]
  if soc:lines.append(f'SoC 温度  {max(soc):.1f} °C')
  if bpu:lines.append(f'BPU 温度  {max(bpu):.1f} °C')
  self.info_text='\n'.join(lines);self.info.set_size_request(480,24+27*len(lines));self.info.queue_draw()
  self.resize_panel()
  (ROOT/'local-display-resources.json').write_text(json.dumps({'resources':r,'frame':s,'panel_visible':self.panel_visible,'cursor_hidden':self.cursor_hidden,'panel_text':'\n'.join(lines)},ensure_ascii=False,indent=2))
  return True
 def key(self,window,event):
  focus=self.window.get_focus()
  if event.keyval==Gdk.KEY_Escape:self.window.unfullscreen()
  elif event.keyval==Gdk.KEY_F11 or (event.keyval==Gdk.KEY_f and not isinstance(focus,Gtk.Entry)):self.window.fullscreen()
  elif event.keyval==Gdk.KEY_space and not isinstance(focus,Gtk.Entry) and self.festival:self.festival.trigger()
 def draw_frame(self,widget,context):
  if self.pixbuf is None:return False
  width=widget.get_allocated_width();height=widget.get_allocated_height();w=self.pixbuf.get_width();h=self.pixbuf.get_height()
  if min(width,height,w,h)<=0:return False
  left,top,scale=cover_geometry(width,height,w,h)
  context.save();context.rectangle(0,0,width,height);context.clip();context.translate(left,top);context.scale(scale,scale)
  Gdk.cairo_set_source_pixbuf(context,self.pixbuf,0,0);context.paint();context.restore()
  if self.countdown:
   layout=PangoCairo.create_layout(context);layout.set_font_description(Pango.FontDescription('Sans Bold 100'));layout.set_text(self.countdown,-1);tw,th=layout.get_pixel_size()
   context.set_source_rgba(0,0,0,.5);context.rectangle((width-tw)/2-24,(height-th)/2-12,tw+48,th+24);context.fill()
   context.set_source_rgb(1,1,1);context.move_to((width-tw)/2,(height-th)/2);PangoCairo.show_layout(context,layout)
  return False
 def draw_panel(self,widget,context):
  context.save();context.set_source_rgba(0,0,0,0.45)
  context.rectangle(0,0,widget.get_allocated_width(),widget.get_allocated_height());context.fill();context.restore()
  return False
 def draw_info(self,widget,context):
  font=Pango.FontDescription('Sans');font.set_absolute_size(19*Pango.SCALE)
  line_layout=PangoCairo.create_layout(context);line_layout.set_font_description(font)
  context.set_source_rgb(1,1,1)
  for row,line in enumerate(self.info_text.splitlines()):
   context.move_to(18,12+row*27);line_layout.set_text(line,-1);PangoCairo.show_layout(context,line_layout)
  return False
 def window_state(self,window,event):
  # The title-bar maximize button should return to kiosk fullscreen too.
  if event.new_window_state&Gdk.WindowState.MAXIMIZED and not event.new_window_state&Gdk.WindowState.FULLSCREEN:
   GLib.idle_add(self.restore_fullscreen)
  (ROOT/'local-display-window.json').write_text(json.dumps({'fullscreen':bool(event.new_window_state&Gdk.WindowState.FULLSCREEN),'time':time.time()}))
 def restore_fullscreen(self):
  self.window.unmaximize();self.window.fullscreen()
  return False
 def capture(self):
  while self.running:
   devices=sorted(Path('/dev/v4l/by-id').glob('*video-index0')) or sorted(Path('/dev').glob('video*'))
   if not devices:time.sleep(1);continue
   cap=None
   for device in devices:
    cap=cv2.VideoCapture(str(device),cv2.CAP_V4L2)
    if cap.isOpened():break
    cap.release();cap=None
   if cap is None:time.sleep(1);continue
   modes=subprocess.run(['v4l2-ctl','-d',str(device),'--list-formats-ext'],capture_output=True,text=True).stdout
   candidates=[];fmt='';size=None
   for line in modes.splitlines():
    m=re.search(r"\[\d+\]: '(.{4})'",line)
    if m:fmt=m[1]
    m=re.search(r'Size: Discrete (\d+)x(\d+)',line)
    if m:size=(int(m[1]),int(m[2]))
    m=re.search(r'\(([\d.]+) fps\)',line)
    if m and size:candidates.append((size[0]*size[1],float(m[1]),fmt,size))
   if candidates:
    _,fps,fmt,(w,h)=max(candidates,key=lambda x:(x[0],x[1],x[2]=='MJPG'))
    if fmt=='MJPG' and os.environ.get('RAC_HWDEC','1')=='1':
     from fast_capture import HwMjpegCapture
     cap.release();cap=HwMjpegCapture(device,w,h,fps)
    else:
     cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*fmt));cap.set(cv2.CAP_PROP_FRAME_WIDTH,w);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,h);cap.set(cv2.CAP_PROP_FPS,fps)
     cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
   self.capture_info={'device':str(device),'width':cap.get(3),'height':cap.get(4),'fps':cap.get(5)}
   print('CAPTURE',json.dumps(self.capture_info),flush=True)
   while self.running:
    ok,frame=cap.read()
    if not ok:break
    with self.lock:self.latest=frame;self.sequence+=1
   cap.release()
   with self.lock:self.latest=None
   self.message='摄像头已断开，等待重新连接…';time.sleep(1)
 def process(self):
  from inference_pipeline import run_pipeline
  count=0;last_settings=None
  def latest():
   with self.lock:return self.sequence,self.latest
  while self.running:
   try:
    for output,metrics in run_pipeline(latest,lambda:self.running,configuration=lambda:self.settings):
     if metrics['settings']!=self.settings:continue
     if metrics['settings']!=last_settings:self.frame_times.clear();last_settings=metrics['settings']
     self.result=output;count+=1
     if self.festival:self.festival.accept(output,metrics['capture_sequence'])
     now=time.monotonic();self.frame_times.append(now)
     while len(self.frame_times)>2 and self.frame_times[0]<now-3:self.frame_times.popleft()
     fps=(len(self.frame_times)-1)/(now-self.frame_times[0]) if len(self.frame_times)>1 else 1000/max(metrics['latency_ms'],1)
     self.message='';status={'frames':count,'fps':fps,'fps_window_seconds':3,'capture':self.capture_info,'output':[output.shape[1],output.shape[0]],'mirror':self.settings['mirror'],'display_mode':'cover','backend':'S100 BPU','pipeline':f"{os.environ.get('RAC_WORKERS','2')}-process-gated-{os.environ.get('RAC_INFLIGHT','1')}"}
     status.update(metrics);self.frame_status=status
     if count%10==1:
      (ROOT/'local-display-status.json').write_text(json.dumps(status,indent=2));print('DISPLAY',json.dumps(status),flush=True)
   except Exception as e:self.message='正在恢复推理进程…';print('INFERENCE',str(e),flush=True);time.sleep(1)
 def paint(self):
  if self.festival:
   countdown=self.festival.tick()
   if countdown!=self.countdown:self.countdown=countdown;self.image.queue_draw()
   self.print_note.set_text(self.festival.status)
  self.label.set_text(self.message);self.label.set_visible(bool(self.message))
  if time.monotonic()-self.heartbeat>2:
   self.heartbeat=time.monotonic();(ROOT/'local-display-heartbeat.json').write_text(json.dumps({'message':self.message,'camera':self.capture_info,'has_output':self.result is not None,'time':time.time()}))
  if self.result is not None:
   rgb=self.result;h,w=rgb.shape[:2]
   if self.painted is rgb:return True
   self.painted=rgb
   self.pixbuf=GdkPixbuf.Pixbuf.new_from_bytes(GLib.Bytes.new(rgb.tobytes()),GdkPixbuf.Colorspace.RGB,False,8,w,h,w*3)
   self.image.queue_draw()
   if self.panel_visible:self.info.queue_draw()
  return True

if __name__=='__main__':
 import signal
 player=Player()
 signal.signal(signal.SIGTERM,lambda *_:GLib.idle_add(player.window.destroy))
 signal.signal(signal.SIGINT,lambda *_:GLib.idle_add(player.window.destroy))
 signal.signal(signal.SIGUSR1,lambda *_:GLib.idle_add(player.festival.trigger))
 try:Gtk.main()
 finally:
  player.running=False
  if hasattr(player,'capture_thread'):player.capture_thread.join(timeout=5)
  if hasattr(player,'processing_thread'):player.processing_thread.join(timeout=10)
  if player.festival:
   player.festival.detector.join(timeout=5);player.festival.worker.join(timeout=35)
