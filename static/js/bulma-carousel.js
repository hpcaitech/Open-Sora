(function webpackUniversalModuleDefinition(root, factory) {
	if(typeof exports === 'object' && typeof module === 'object')
		module.exports = factory();
	else if(typeof define === 'function' && define.amd)
		define([], factory);
	else if(typeof exports === 'object')
		exports["bulmaCarousel"] = factory();
	else
		root["bulmaCarousel"] = factory();
})(typeof self !== 'undefined' ? self : this, function() {
return /******/ (function(modules) { // webpackBootstrap
/******/ 	// The module cache
/******/ 	var installedModules = {};
/******/
/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {
/******/
/******/ 		// Check if module is in cache
/******/ 		if(installedModules[moduleId]) {
/******/ 			return installedModules[moduleId].exports;
/******/ 		}
/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = installedModules[moduleId] = {
/******/ 			i: moduleId,
/******/ 			l: false,
/******/ 			exports: {}
/******/ 		};
/******/
/******/ 		// Execute the module function
/******/ 		modules[moduleId].call(module.exports, module, module.exports, __webpack_require__);
/******/
/******/ 		// Flag the module as loaded
/******/ 		module.l = true;
/******/
/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}
/******/
/******/
/******/ 	// expose the modules object (__webpack_modules__)
/******/ 	__webpack_require__.m = modules;
/******/
/******/ 	// expose the module cache
/******/ 	__webpack_require__.c = installedModules;
/******/
/******/ 	// define getter function for harmony exports
/******/ 	__webpack_require__.d = function(exports, name, getter) {
/******/ 		if(!__webpack_require__.o(exports, name)) {
/******/ 			Object.defineProperty(exports, name, {
/******/ 				configurable: false,
/******/ 				enumerable: true,
/******/ 				get: getter
/******/ 			});
/******/ 		}
/******/ 	};
/******/
/******/ 	// getDefaultExport function for compatibility with non-harmony modules
/******/ 	__webpack_require__.n = function(module) {
/******/ 		var getter = module && module.__esModule ?
/******/ 			function getDefault() { return module['default']; } :
/******/ 			function getModuleExports() { return module; };
/******/ 		__webpack_require__.d(getter, 'a', getter);
/******/ 		return getter;
/******/ 	};
/******/
/******/ 	// Object.prototype.hasOwnProperty.call
/******/ 	__webpack_require__.o = function(object, property) { return Object.prototype.hasOwnProperty.call(object, property); };
/******/
/******/ 	// __webpack_public_path__
/******/ 	__webpack_require__.p = "";
/******/
/******/ 	// Load entry module and return exports
/******/ 	return __webpack_require__(__webpack_require__.s = 5);
/******/ })
/************************************************************************/
/******/ ([
/* 0 */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
/* unused harmony export addClasses */
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "d", function() { return removeClasses; });
/* unused harmony export show */
/* unused harmony export hide */
/* unused harmony export offset */
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "e", function() { return width; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "b", function() { return height; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "c", function() { return outerHeight; });
/* unused harmony export outerWidth */
/* unused harmony export position */
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "a", function() { return css; });
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_0__type__ = __webpack_require__(2);


var addClasses = function addClasses(element, classes) {
	classes = Array.isArray(classes) ? classes : classes.split(' ');
	classes.forEach(function (cls) {
		element.classList.add(cls);
	});
};

var removeClasses = function removeClasses(element, classes) {
	classes = Array.isArray(classes) ? classes : classes.split(' ');
	classes.forEach(function (cls) {
		element.classList.remove(cls);
	});
};

var show = function show(elements) {
	elements = Array.isArray(elements) ? elements : [elements];
	elements.forEach(function (element) {
		element.style.display = '';
	});
};

var hide = function hide(elements) {
	elements = Array.isArray(elements) ? elements : [elements];
	elements.forEach(function (element) {
		element.style.display = 'none';
	});
};

var offset = function offset(element) {
	var rect = element.getBoundingClientRect();
	return {
		top: rect.top + document.body.scrollTop,
		left: rect.left + document.body.scrollLeft
	};
};

// returns an element's width
var width = function width(element) {
	return element.getBoundingClientRect().width || element.offsetWidth;
};
// returns an element's height
var height = function height(element) {
	return element.getBoundingClientRect().height || element.offsetHeight;
};

var outerHeight = function outerHeight(element) {
	var withMargin = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : false;

	var height = element.offsetHeight;
	if (withMargin) {
		var style = window.getComputedStyle(element);
		height += parseInt(style.marginTop) + parseInt(style.marginBottom);
	}
	return height;
};

var outerWidth = function outerWidth(element) {
	var withMargin = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : false;

	var width = element.offsetWidth;
	if (withMargin) {
		var style = window.getComputedStyle(element);
		width += parseInt(style.marginLeft) + parseInt(style.marginRight);
	}
	return width;
};

var position = function position(element) {
	return {
		left: element.offsetLeft,
		top: element.offsetTop
	};
};

var css = function css(element, obj) {
	if (!obj) {
		return window.getComputedStyle(element);
	}
	if (Object(__WEBPACK_IMPORTED_MODULE_0__type__["b" /* isObject */])(obj)) {
		var style = '';
		Object.keys(obj).forEach(function (key) {
			style += key + ': ' + obj[key] + ';';
		});

		element.style.cssText += style;
	}
};

/***/ }),
/* 1 */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
/* harmony export (immutable) */ __webpack_exports__["a"] = detectSupportsPassive;
function detectSupportsPassive() {
	var supportsPassive = false;

	try {
		var opts = Object.defineProperty({}, 'passive', {
			get: function get() {
				supportsPassive = true;
			}
		});

		window.addEventListener('testPassive', null, opts);
		window.removeEventListener('testPassive', null, opts);
	} catch (e) {}

	return supportsPassive;
}

/***/ }),
/* 2 */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "a", function() { return isFunction; });
/* unused harmony export isNumber */
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "c", function() { return isString; });
/* unused harmony export isDate */
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "b", function() { return isObject; });
/* unused harmony export isEmptyObject */
/* unused harmony export isNode */
/* unused harmony export isVideo */
/* unused harmony export isHTML5 */
/* unused harmony export isIFrame */
/* unused harmony export isYoutube */
/* unused harmony export isVimeo */
var _typeof = typeof Symbol === "function" && typeof Symbol.iterator === "symbol" ? function (obj) { return typeof obj; } : function (obj) { return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; };

var isFunction = function isFunction(unknown) {
	return typeof unknown === 'function';
};
var isNumber = function isNumber(unknown) {
	return typeof unknown === "number";
};
var isString = function isString(unknown) {
	return typeof unknown === 'string' || !!unknown && (typeof unknown === 'undefined' ? 'undefined' : _typeof(unknown)) === 'object' && Object.prototype.toString.call(unknown) === '[object String]';
};
var isDate = function isDate(unknown) {
	return (Object.prototype.toString.call(unknown) === '[object Date]' || unknown instanceof Date) && !isNaN(unknown.valueOf());
};
var isObject = function isObject(unknown) {
	return (typeof unknown === 'function' || (typeof unknown === 'undefined' ? 'undefined' : _typeof(unknown)) === 'object' && !!unknown) && !Array.isArray(unknown);
};
var isEmptyObject = function isEmptyObject(unknown) {
	for (var name in unknown) {
		if (unknown.hasOwnProperty(name)) {
			return false;
		}
	}
	return true;
};

var isNode = function isNode(unknown) {
	return !!(unknown && unknown.nodeType === HTMLElement | SVGElement);
};
var isVideo = function isVideo(unknown) {
	return isYoutube(unknown) || isVimeo(unknown) || isHTML5(unknown);
};
var isHTML5 = function isHTML5(unknown) {
	return isNode(unknown) && unknown.tagName === 'VIDEO';
};
var isIFrame = function isIFrame(unknown) {
	return isNode(unknown) && unknown.tagName === 'IFRAME';
};
var isYoutube = function isYoutube(unknown) {
	return isIFrame(unknown) && !!unknown.src.match(/\/\/.*?youtube(-nocookie)?\.[a-z]+\/(watch\?v=[^&\s]+|embed)|youtu\.be\/.*/);
};
var isVimeo = function isVimeo(unknown) {
	return isIFrame(unknown) && !!unknown.src.match(/vimeo\.com\/video\/.*/);
};

/***/ }),
/* 3 */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _toConsumableArray(arr) { if (Array.isArray(arr)) { for (var i = 0, arr2 = Array(arr.length); i < arr.length; i++) { arr2[i] = arr[i]; } return arr2; } else { return Array.from(arr); } }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var EventEmitter = function () {
  function EventEmitter() {
    var events = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : [];

    _classCallCheck(this, EventEmitter);

    this.events = new Map(events);
  }

  _createClass(EventEmitter, [{
    key: "on",
    value: function on(name, cb) {
      var _this = this;

      this.events.set(name, [].concat(_toConsumableArray(this.events.has(name) ? this.events.get(name) : []), [cb]));

      return function () {
        return _this.events.set(name, _this.events.get(name).filter(function (fn) {
          return fn !== cb;
        }));
      };
    }
  }, {
    key: "emit",
    value: function emit(name) {
      for (var _len = arguments.length, args = Array(_len > 1 ? _len - 1 : 0), _key = 1; _key < _len; _key++) {
        args[_key - 1] = arguments[_key];
      }

      return this.events.has(name) && this.events.get(name).map(function (fn) {
        return fn.apply(undefined, args);
      });
    }
  }]);

  return EventEmitter;
}();

/* harmony default export */ __webpack_exports__["a"] = (EventEmitter);

/***/ }),
/* 4 */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Coordinate = function () {
	function Coordinate() {
		var x = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : 0;
		var y = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 0;

		_classCallCheck(this, Coordinate);

		this._x = x;
		this._y = y;
	}

	_createClass(Coordinate, [{
		key: 'add',
		value: function add(coord) {
			return new Coordinate(this._x + coord._x, this._y + coord._y);
		}
	}, {
		key: 'sub',
		value: function sub(coord) {
			return new Coordinate(this._x - coord._x, this._y - coord._y);
		}
	}, {
		key: 'distance',
		value: function distance(coord) {
			var deltaX = this._x - coord._x;
			var deltaY = this._y - coord._y;

			return Math.sqrt(Math.pow(deltaX, 2) + Math.pow(deltaY, 2));
		}
	}, {
		key: 'max',
		value: function max(coord) {
			var x = Math.max(this._x, coord._x);
			var y = Math.max(this._y, coord._y);

			return new Coordinate(x, y);
		}
	}, {
		key: 'equals',
		value: function equals(coord) {
			if (this == coord) {
				return true;
			}
			if (!coord || coord == null) {
				return false;
			}
			return this._x == coord._x && this._y == coord._y;
		}
	}, {
		key: 'inside',
		value: function inside(northwest, southeast) {
			if (this._x >= northwest._x && this._x <= southeast._x && this._y >= northwest._y && this._y <= southeast._y) {

				return true;
			}
			return false;
		}
	}, {
		key: 'constrain',
		value: function constrain(min, max) {
			if (min._x > max._x || min._y > max._y) {
				return this;
			}

			var x = this._x,
			    y = this._y;

			if (min._x !== null) {
				x = Math.max(x, min._x);
			}
			if (max._x !== null) {
				x = Math.min(x, max._x);
			}
			if (min._y !== null) {
				y = Math.max(y, min._y);
			}
			if (max._y !== null) {
				y = Math.min(y, max._y);
			}

			return new Coordinate(x, y);
		}
	}, {
		key: 'reposition',
		value: function reposition(element) {
			element.style['top'] = this._y + 'px';
			element.style['left'] = this._x + 'px';
		}
	}, {
		key: 'toString',
		value: function toString() {
			return '(' + this._x + ',' + this._y + ')';
		}
	}, {
		key: 'x',
		get: function get() {
			return this._x;
		},
		set: function set() {
			var value = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : 0;

			this._x = value;
			return this;
		}
	}, {
		key: 'y',
		get: function get() {
			return this._y;
		},
		set: function set() {
			var value = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : 0;

			this._y = value;
			return this;
		}
	}]);

	return Coordinate;
}();

/* harmony default export */ __webpack_exports__["a"] = (Coordinate);

/***/ }),
/* 5 */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
Object.defineProperty(__webpack_exports__, "__esModule", { value: true });
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_0__utils_index__ = __webpack_require__(6);
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_1__utils_css__ = __webpack_require__(0);
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_2__utils_type__ = __webpack_require__(2);
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_3__utils_eventEmitter__ = __webpack_require__(3);
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_4__components_autoplay__ = __webpack_require__(7);
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_5__components_breakpoint__ = __webpack_require__(9);
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_6__components_infinite__ = __webpack_require__(10);
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_7__components_loop__ = __webpack_require__(11);
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_8__components_navigation__ = __webpack_require__(13);
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_9__components_pagination__ = __webpack_require__(15);
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_10__components_swipe__ = __webpack_require__(18);
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_11__components_transitioner__ = __webpack_require__(19);
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_12__defaultOptions__ = __webpack_require__(22);
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_13__templates__ = __webpack_require__(23);
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_14__templates_item__ = __webpack_require__(24);
var _extends = Object.assign || function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; };

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _possibleConstructorReturn(self, call) { if (!self) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return call && (typeof call === "object" || typeof call === "function") ? call : self; }

function _inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function, not " + typeof superClass); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, enumerable: false, writable: true, configurable: true } }); if (superClass) Object.setPrototypeOf ? Object.setPrototypeOf(subClass, superClass) : subClass.__proto__ = superClass; }



















var bulmaCarousel = function (_EventEmitter) {
  _inherits(bulmaCarousel, _EventEmitter);

  function bulmaCarousel(selector) {
    var options = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};

    _classCallCheck(this, bulmaCarousel);

    var _this = _possibleConstructorReturn(this, (bulmaCarousel.__proto__ || Object.getPrototypeOf(bulmaCarousel)).call(this));

    _this.element = Object(__WEBPACK_IMPORTED_MODULE_2__utils_type__["c" /* isString */])(selector) ? document.querySelector(selector) : selector;
    // An invalid selector or non-DOM node has been provided.
    if (!_this.element) {
      throw new Error('An invalid selector or non-DOM node has been provided.');
    }
    _this._clickEvents = ['click', 'touch'];

    // Use Element dataset values to override options
    var elementConfig = _this.element.dataset ? Object.keys(_this.element.dataset).filter(function (key) {
      return Object.keys(__WEBPACK_IMPORTED_MODULE_12__defaultOptions__["a" /* default */]).includes(key);
    }).reduce(function (obj, key) {
      return _extends({}, obj, _defineProperty({}, key, _this.element.dataset[key]));
    }, {}) : {};
    // Set default options - dataset attributes are master
    _this.options = _extends({}, __WEBPACK_IMPORTED_MODULE_12__defaultOptions__["a" /* default */], options, elementConfig);

    _this._id = Object(__WEBPACK_IMPORTED_MODULE_0__utils_index__["a" /* uuid */])('slider');

    _this.onShow = _this.onShow.bind(_this);

    // Initiate plugin
    _this._init();
    return _this;
  }

  /**
   * Initiate all DOM element containing datePicker class
   * @method
   * @return {Array} Array of all datePicker instances
   */


  _createClass(bulmaCarousel, [{
    key: '_init',


    /****************************************************
     *                                                  *
     * PRIVATE FUNCTIONS                                *
     *                                                  *
     ****************************************************/
    /**
     * Initiate plugin instance
     * @method _init
     * @return {Slider} Current plugin instance
     */
    value: function _init() {
      this._items = Array.from(this.element.children);

      // Load plugins
      this._breakpoint = new __WEBPACK_IMPORTED_MODULE_5__components_breakpoint__["a" /* default */](this);
      this._autoplay = new __WEBPACK_IMPORTED_MODULE_4__components_autoplay__["a" /* default */](this);
      this._navigation = new __WEBPACK_IMPORTED_MODULE_8__components_navigation__["a" /* default */](this);
      this._pagination = new __WEBPACK_IMPORTED_MODULE_9__components_pagination__["a" /* default */](this);
      this._infinite = new __WEBPACK_IMPORTED_MODULE_6__components_infinite__["a" /* default */](this);
      this._loop = new __WEBPACK_IMPORTED_MODULE_7__components_loop__["a" /* default */](this);
      this._swipe = new __WEBPACK_IMPORTED_MODULE_10__components_swipe__["a" /* default */](this);

      this._build();

      if (Object(__WEBPACK_IMPORTED_MODULE_2__utils_type__["a" /* isFunction */])(this.options.onReady)) {
        this.options.onReady(this);
      }

      return this;
    }

    /**
     * Build Slider HTML component and append it to the DOM
     * @method _build
     */

  }, {
    key: '_build',
    value: function _build() {
      var _this2 = this;

      // Generate HTML Fragment of template
      this.node = document.createRange().createContextualFragment(Object(__WEBPACK_IMPORTED_MODULE_13__templates__["a" /* default */])(this.id));
      // Save pointers to template parts
      this._ui = {
        wrapper: this.node.firstChild,
        container: this.node.querySelector('.slider-container')

        // Add slider to DOM
      };this.element.appendChild(this.node);
      this._ui.wrapper.classList.add('is-loading');
      this._ui.container.style.opacity = 0;

      this._transitioner = new __WEBPACK_IMPORTED_MODULE_11__components_transitioner__["a" /* default */](this);

      // Wrap all items by slide element
      this._slides = this._items.map(function (item, index) {
        return _this2._createSlide(item, index);
      });

      this.reset();

      this._bindEvents();

      this._ui.container.style.opacity = 1;
      this._ui.wrapper.classList.remove('is-loading');
    }

    /**
     * Bind all events
     * @method _bindEvents
     * @return {void}
     */

  }, {
    key: '_bindEvents',
    value: function _bindEvents() {
      this.on('show', this.onShow);
    }
  }, {
    key: '_unbindEvents',
    value: function _unbindEvents() {
      this.off('show', this.onShow);
    }
  }, {
    key: '_createSlide',
    value: function _createSlide(item, index) {
      var slide = document.createRange().createContextualFragment(Object(__WEBPACK_IMPORTED_MODULE_14__templates_item__["a" /* default */])()).firstChild;
      slide.dataset.sliderIndex = index;
      slide.appendChild(item);
      return slide;
    }

    /**
     * Calculate slider dimensions
     */

  }, {
    key: '_setDimensions',
    value: function _setDimensions() {
      var _this3 = this;

      if (!this.options.vertical) {
        if (this.options.centerMode) {
          this._ui.wrapper.style.padding = '0px ' + this.options.centerPadding;
        }
      } else {
        this._ui.wrapper.style.height = Object(__WEBPACK_IMPORTED_MODULE_1__utils_css__["c" /* outerHeight */])(this._slides[0]) * this.slidesToShow;
        if (this.options.centerMode) {
          this._ui.wrapper.style.padding = this.options.centerPadding + ' 0px';
        }
      }

      this._wrapperWidth = Object(__WEBPACK_IMPORTED_MODULE_1__utils_css__["e" /* width */])(this._ui.wrapper);
      this._wrapperHeight = Object(__WEBPACK_IMPORTED_MODULE_1__utils_css__["c" /* outerHeight */])(this._ui.wrapper);

      if (!this.options.vertical) {
        this._slideWidth = Math.ceil(this._wrapperWidth / this.slidesToShow);
        this._containerWidth = Math.ceil(this._slideWidth * this._slides.length);
        this._ui.container.style.width = this._containerWidth + 'px';
      } else {
        this._slideWidth = Math.ceil(this._wrapperWidth);
        this._containerHeight = Math.ceil(Object(__WEBPACK_IMPORTED_MODULE_1__utils_css__["c" /* outerHeight */])(this._slides[0]) * this._slides.length);
        this._ui.container.style.height = this._containerHeight + 'px';
      }

      this._slides.forEach(function (slide) {
        slide.style.width = _this3._slideWidth + 'px';
      });
    }
  }, {
    key: '_setHeight',
    value: function _setHeight() {
      if (this.options.effect !== 'translate') {
        this._ui.container.style.height = Object(__WEBPACK_IMPORTED_MODULE_1__utils_css__["c" /* outerHeight */])(this._slides[this.state.index]) + 'px';
      }
    }

    // Update slides classes

  }, {
    key: '_setClasses',
    value: function _setClasses() {
      var _this4 = this;

      this._slides.forEach(function (slide) {
        Object(__WEBPACK_IMPORTED_MODULE_1__utils_css__["d" /* removeClasses */])(slide, 'is-active is-current is-slide-previous is-slide-next');
        if (Math.abs((_this4.state.index - 1) % _this4.state.length) === parseInt(slide.dataset.sliderIndex, 10)) {
          slide.classList.add('is-slide-previous');
        }
        if (Math.abs(_this4.state.index % _this4.state.length) === parseInt(slide.dataset.sliderIndex, 10)) {
          slide.classList.add('is-current');
        }
        if (Math.abs((_this4.state.index + 1) % _this4.state.length) === parseInt(slide.dataset.sliderIndex, 10)) {
          slide.classList.add('is-slide-next');
        }
      });
    }

    /****************************************************
     *                                                  *
     * GETTERS and SETTERS                              *
     *                                                  *
     ****************************************************/

    /**
     * Get id of current datePicker
     */

  }, {
    key: 'onShow',


    /****************************************************
     *                                                  *
     * EVENTS FUNCTIONS                                 *
     *                                                  *
     ****************************************************/
    value: function onShow(e) {
      this._navigation.refresh();
      this._pagination.refresh();
      this._setClasses();
    }

    /****************************************************
     *                                                  *
     * PUBLIC FUNCTIONS                                 *
     *                                                  *
     ****************************************************/

  }, {
    key: 'next',
    value: function next() {
      if (!this.options.loop && !this.options.infinite && this.state.index + this.slidesToScroll > this.state.length - this.slidesToShow && !this.options.centerMode) {
        this.state.next = this.state.index;
      } else {
        this.state.next = this.state.index + this.slidesToScroll;
      }
      this.show();
    }
  }, {
    key: 'previous',
    value: function previous() {
      if (!this.options.loop && !this.options.infinite && this.state.index === 0) {
        this.state.next = this.state.index;
      } else {
        this.state.next = this.state.index - this.slidesToScroll;
      }
      this.show();
    }
  }, {
    key: 'start',
    value: function start() {
      this._autoplay.start();
    }
  }, {
    key: 'pause',
    value: function pause() {
      this._autoplay.pause();
    }
  }, {
    key: 'stop',
    value: function stop() {
      this._autoplay.stop();
    }
  }, {
    key: 'show',
    value: function show(index) {
      var force = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : false;

      // If all slides are already visible then return
      if (!this.state.length || this.state.length <= this.slidesToShow) {
        return;
      }

      if (typeof index === 'Number') {
        this.state.next = index;
      }

      if (this.options.loop) {
        this._loop.apply();
      }
      if (this.options.infinite) {
        this._infinite.apply();
      }

      // If new slide is already the current one then return
      if (this.state.index === this.state.next) {
        return;
      }

      this.emit('before:show', this.state);
      this._transitioner.apply(force, this._setHeight.bind(this));
      this.emit('after:show', this.state);

      this.emit('show', this);
    }
  }, {
    key: 'reset',
    value: function reset() {
      var _this5 = this;

      this.state = {
        length: this._items.length,
        index: Math.abs(this.options.initialSlide),
        next: Math.abs(this.options.initialSlide),
        prev: undefined
      };

      // Fix options
      if (this.options.loop && this.options.infinite) {
        this.options.loop = false;
      }
      if (this.options.slidesToScroll > this.options.slidesToShow) {
        this.options.slidesToScroll = this.slidesToShow;
      }
      this._breakpoint.init();

      if (this.state.index >= this.state.length && this.state.index !== 0) {
        this.state.index = this.state.index - this.slidesToScroll;
      }
      if (this.state.length <= this.slidesToShow) {
        this.state.index = 0;
      }

      this._ui.wrapper.appendChild(this._navigation.init().render());
      this._ui.wrapper.appendChild(this._pagination.init().render());

      if (this.options.navigationSwipe) {
        this._swipe.bindEvents();
      } else {
        this._swipe._bindEvents();
      }

      this._breakpoint.apply();
      // Move all created slides into slider
      this._slides.forEach(function (slide) {
        return _this5._ui.container.appendChild(slide);
      });
      this._transitioner.init().apply(true, this._setHeight.bind(this));

      if (this.options.autoplay) {
        this._autoplay.init().start();
      }
    }

    /**
     * Destroy Slider
     * @method destroy
     */

  }, {
    key: 'destroy',
    value: function destroy() {
      var _this6 = this;

      this._unbindEvents();
      this._items.forEach(function (item) {
        _this6.element.appendChild(item);
      });
      this.node.remove();
    }
  }, {
    key: 'id',
    get: function get() {
      return this._id;
    }
  }, {
    key: 'index',
    set: function set(index) {
      this._index = index;
    },
    get: function get() {
      return this._index;
    }
  }, {
    key: 'length',
    set: function set(length) {
      this._length = length;
    },
    get: function get() {
      return this._length;
    }
  }, {
    key: 'slides',
    get: function get() {
      return this._slides;
    },
    set: function set(slides) {
      this._slides = slides;
    }
  }, {
    key: 'slidesToScroll',
    get: function get() {
      return this.options.effect === 'translate' ? this._breakpoint.getSlidesToScroll() : 1;
    }
  }, {
    key: 'slidesToShow',
    get: function get() {
      return this.options.effect === 'translate' ? this._breakpoint.getSlidesToShow() : 1;
    }
  }, {
    key: 'direction',
    get: function get() {
      return this.element.dir.toLowerCase() === 'rtl' || this.element.style.direction === 'rtl' ? 'rtl' : 'ltr';
    }
  }, {
    key: 'wrapper',
    get: function get() {
      return this._ui.wrapper;
    }
  }, {
    key: 'wrapperWidth',
    get: function get() {
      return this._wrapperWidth || 0;
    }
  }, {
    key: 'container',
    get: function get() {
      return this._ui.container;
    }
  }, {
    key: 'containerWidth',
    get: function get() {
      return this._containerWidth || 0;
    }
  }, {
    key: 'slideWidth',
    get: function get() {
      return this._slideWidth || 0;
    }
  }, {
    key: 'transitioner',
    get: function get() {
      return this._transitioner;
    }
  }], [{
    key: 'attach',
    value: function attach() {
      var _this7 = this;

      var selector = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : '.slider';
      var options = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};

      var instances = new Array();

      var elements = Object(__WEBPACK_IMPORTED_MODULE_2__utils_type__["c" /* isString */])(selector) ? document.querySelectorAll(selector) : Array.isArray(selector) ? selector : [selector];
      [].forEach.call(elements, function (element) {
        if (typeof element[_this7.constructor.name] === 'undefined') {
          var instance = new bulmaCarousel(element, options);
          element[_this7.constructor.name] = instance;
          instances.push(instance);
        } else {
          instances.push(element[_this7.constructor.name]);
        }
      });

      return instances;
    }
  }]);

  return bulmaCarousel;
}(__WEBPACK_IMPORTED_MODULE_3__utils_eventEmitter__["a" /* default */]);

/* harmony default export */ __webpack_exports__["default"] = (bulmaCarousel);

/***/ }),
/* 6 */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "a", function() { return uuid; });
/* unused harmony export isRtl */
/* unused harmony export defer */
/* unused harmony export getNodeIndex */
/* unused harmony export camelize */
function _toConsumableArray(arr) { if (Array.isArray(arr)) { for (var i = 0, arr2 = Array(arr.length); i < arr.length; i++) { arr2[i] = arr[i]; } return arr2; } else { return Array.from(arr); } }

var uuid = function uuid() {
	var prefix = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : '';
	return prefix + ([1e7] + -1e3 + -4e3 + -8e3 + -1e11).replace(/[018]/g, function (c) {
		return (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16);
	});
};
var isRtl = function isRtl() {
	return document.documentElement.getAttribute('dir') === 'rtl';
};

var defer = function defer() {
	this.promise = new Promise(function (resolve, reject) {
		this.resolve = resolve;
		this.reject = reject;
	}.bind(this));

	this.then = this.promise.then.bind(this.promise);
	this.catch = this.promise.catch.bind(this.promise);
};

var getNodeIndex = function getNodeIndex(node) {
	return [].concat(_toConsumableArray(node.parentNode.children)).indexOf(node);
};
var camelize = function camelize(str) {
	return str.replace(/-(\w)/g, toUpper);
};

/***/ }),
/* 7 */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_0__utils_eventEmitter__ = __webpack_require__(3);
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_1__utils_device__ = __webpack_require__(8);
var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _possibleConstructorReturn(self, call) { if (!self) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return call && (typeof call === "object" || typeof call === "function") ? call : self; }

function _inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function, not " + typeof superClass); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, enumerable: false, writable: true, configurable: true } }); if (superClass) Object.setPrototypeOf ? Object.setPrototypeOf(subClass, superClass) : subClass.__proto__ = superClass; }




var onVisibilityChange = Symbol('onVisibilityChange');
var onMouseEnter = Symbol('onMouseEnter');
var onMouseLeave = Symbol('onMouseLeave');

var defaultOptions = {
	autoplay: false,
	autoplaySpeed: 3000
};

var Autoplay = function (_EventEmitter) {
	_inherits(Autoplay, _EventEmitter);

	function Autoplay(slider) {
		_classCallCheck(this, Autoplay);

		var _this = _possibleConstructorReturn(this, (Autoplay.__proto__ || Object.getPrototypeOf(Autoplay)).call(this));

		_this.slider = slider;

		_this.onVisibilityChange = _this.onVisibilityChange.bind(_this);
		_this.onMouseEnter = _this.onMouseEnter.bind(_this);
		_this.onMouseLeave = _this.onMouseLeave.bind(_this);
		return _this;
	}

	_createClass(Autoplay, [{
		key: 'init',
		value: function init() {
			this._bindEvents();
			return this;
		}
	}, {
		key: '_bindEvents',
		value: function _bindEvents() {
			document.addEventListener('visibilitychange', this.onVisibilityChange);
			if (this.slider.options.pauseOnHover) {
				this.slider.container.addEventListener(__WEBPACK_IMPORTED_MODULE_1__utils_device__["a" /* pointerEnter */], this.onMouseEnter);
				this.slider.container.addEventListener(__WEBPACK_IMPORTED_MODULE_1__utils_device__["b" /* pointerLeave */], this.onMouseLeave);
			}
		}
	}, {
		key: '_unbindEvents',
		value: function _unbindEvents() {
			document.removeEventListener('visibilitychange', this.onVisibilityChange);
			this.slider.container.removeEventListener(__WEBPACK_IMPORTED_MODULE_1__utils_device__["a" /* pointerEnter */], this.onMouseEnter);
			this.slider.container.removeEventListener(__WEBPACK_IMPORTED_MODULE_1__utils_device__["b" /* pointerLeave */], this.onMouseLeave);
		}
	}, {
		key: 'start',
		value: function start() {
			var _this2 = this;

			this.stop();
			if (this.slider.options.autoplay) {
				this.emit('start', this);
				this._interval = setInterval(function () {
					if (!(_this2._hovering && _this2.slider.options.pauseOnHover)) {
						if (!_this2.slider.options.centerMode && _this2.slider.state.next >= _this2.slider.state.length - _this2.slider.slidesToShow && !_this2.slider.options.loop && !_this2.slider.options.infinite) {
							_this2.stop();
						} else {
							_this2.slider.next();
						}
					}
				}, this.slider.options.autoplaySpeed);
			}
		}
	}, {
		key: 'stop',
		value: function stop() {
			this._interval = clearInterval(this._interval);
			this.emit('stop', this);
		}
	}, {
		key: 'pause',
		value: function pause() {
			var _this3 = this;

			var speed = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : 0;

			if (this.paused) {
				return;
			}
			if (this.timer) {
				this.stop();
			}
			this.paused = true;
			if (speed === 0) {
				this.paused = false;
				this.start();
			} else {
				this.slider.on('transition:end', function () {
					if (!_this3) {
						return;
					}
					_this3.paused = false;
					if (!_this3.run) {
						_this3.stop();
					} else {
						_this3.start();
					}
				});
			}
		}
	}, {
		key: 'onVisibilityChange',
		value: function onVisibilityChange(e) {
			if (document.hidden) {
				this.stop();
			} else {
				this.start();
			}
		}
	}, {
		key: 'onMouseEnter',
		value: function onMouseEnter(e) {
			this._hovering = true;
			if (this.slider.options.pauseOnHover) {
				this.pause();
			}
		}
	}, {
		key: 'onMouseLeave',
		value: function onMouseLeave(e) {
			this._hovering = false;
			if (this.slider.options.pauseOnHover) {
				this.pause();
			}
		}
	}]);

	return Autoplay;
}(__WEBPACK_IMPORTED_MODULE_0__utils_eventEmitter__["a" /* default */]);

/* harmony default export */ __webpack_exports__["a"] = (Autoplay);

/***/ }),
/* 8 */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
/* unused harmony export isIE */
/* unused harmony export isIETouch */
/* unused harmony export isAndroid */
/* unused harmony export isiPad */
/* unused harmony export isiPod */
/* unused harmony export isiPhone */
/* unused harmony export isSafari */
/* unused harmony export isUiWebView */
/* unused harmony export supportsTouchEvents */
/* unused harmony export supportsPointerEvents */
/* unused harmony export supportsTouch */
/* unused harmony export pointerDown */
/* unused harmony export pointerMove */
/* unused harmony export pointerUp */
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "a", function() { return pointerEnter; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "b", function() { return pointerLeave; });
var isIE = window.navigator.pointerEnabled || window.navigator.msPointerEnabled;
var isIETouch = window.navigator.msPointerEnabled && window.navigator.msMaxTouchPoints > 1 || window.navigator.pointerEnabled && window.navigator.maxTouchPoints > 1;
var isAndroid = navigator.userAgent.match(/(Android);?[\s\/]+([\d.]+)?/);
var isiPad = navigator.userAgent.match(/(iPad).*OS\s([\d_]+)/);
var isiPod = navigator.userAgent.match(/(iPod)(.*OS\s([\d_]+))?/);
var isiPhone = !navigator.userAgent.match(/(iPad).*OS\s([\d_]+)/) && navigator.userAgent.match(/(iPhone\sOS)\s([\d_]+)/);
var isSafari = navigator.userAgent.toLowerCase().indexOf('safari') >= 0 && navigator.userAgent.toLowerCase().indexOf('chrome') < 0 && navigator.userAgent.toLowerCase().indexOf('android') < 0;
var isUiWebView = /(iPhone|iPod|iPad).*AppleWebKit(?!.*Safari)/i.test(navigator.userAgent);

var supportsTouchEvents = !!('ontouchstart' in window);
var supportsPointerEvents = !!('PointerEvent' in window);
var supportsTouch = supportsTouchEvents || window.DocumentTouch && document instanceof DocumentTouch || navigator.maxTouchPoints; // IE >=11
var pointerDown = !supportsTouch ? 'mousedown' : 'mousedown ' + (supportsTouchEvents ? 'touchstart' : 'pointerdown');
var pointerMove = !supportsTouch ? 'mousemove' : 'mousemove ' + (supportsTouchEvents ? 'touchmove' : 'pointermove');
var pointerUp = !supportsTouch ? 'mouseup' : 'mouseup ' + (supportsTouchEvents ? 'touchend' : 'pointerup');
var pointerEnter = supportsTouch && supportsPointerEvents ? 'pointerenter' : 'mouseenter';
var pointerLeave = supportsTouch && supportsPointerEvents ? 'pointerleave' : 'mouseleave';

/***/ }),
/* 9 */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var onResize = Symbol('onResize');

var Breakpoints = function () {
	function Breakpoints(slider) {
		_classCallCheck(this, Breakpoints);

		this.slider = slider;
		this.options = slider.options;

		this[onResize] = this[onResize].bind(this);

		this._bindEvents();
	}

	_createClass(Breakpoints, [{
		key: 'init',
		value: function init() {
			this._defaultBreakpoint = {
				slidesToShow: this.options.slidesToShow,
				slidesToScroll: this.options.slidesToScroll
			};
			this.options.breakpoints.sort(function (a, b) {
				return parseInt(a.changePoint, 10) > parseInt(b.changePoint, 10);
			});
			this._currentBreakpoint = this._getActiveBreakpoint();

			return this;
		}
	}, {
		key: 'destroy',
		value: function destroy() {
			this._unbindEvents();
		}
	}, {
		key: '_bindEvents',
		value: function _bindEvents() {
			window.addEventListener('resize', this[onResize]);
			window.addEventListener('orientationchange', this[onResize]);
		}
	}, {
		key: '_unbindEvents',
		value: function _unbindEvents() {
			window.removeEventListener('resize', this[onResize]);
			window.removeEventListener('orientationchange', this[onResize]);
		}
	}, {
		key: '_getActiveBreakpoint',
		value: function _getActiveBreakpoint() {
			//Get breakpoint for window width
			var _iteratorNormalCompletion = true;
			var _didIteratorError = false;
			var _iteratorError = undefined;

			try {
				for (var _iterator = this.options.breakpoints[Symbol.iterator](), _step; !(_iteratorNormalCompletion = (_step = _iterator.next()).done); _iteratorNormalCompletion = true) {
					var point = _step.value;

					if (point.changePoint >= window.innerWidth) {
						return point;
					}
				}
			} catch (err) {
				_didIteratorError = true;
				_iteratorError = err;
			} finally {
				try {
					if (!_iteratorNormalCompletion && _iterator.return) {
						_iterator.return();
					}
				} finally {
					if (_didIteratorError) {
						throw _iteratorError;
					}
				}
			}

			return this._defaultBreakpoint;
		}
	}, {
		key: 'getSlidesToShow',
		value: function getSlidesToShow() {
			return this._currentBreakpoint ? this._currentBreakpoint.slidesToShow : this._defaultBreakpoint.slidesToShow;
		}
	}, {
		key: 'getSlidesToScroll',
		value: function getSlidesToScroll() {
			return this._currentBreakpoint ? this._currentBreakpoint.slidesToScroll : this._defaultBreakpoint.slidesToScroll;
		}
	}, {
		key: 'apply',
		value: function apply() {
			if (this.slider.state.index >= this.slider.state.length && this.slider.state.index !== 0) {
				this.slider.state.index = this.slider.state.index - this._currentBreakpoint.slidesToScroll;
			}
			if (this.slider.state.length <= this._currentBreakpoint.slidesToShow) {
				this.slider.state.index = 0;
			}

			if (this.options.loop) {
				this.slider._loop.init().apply();
			}

			if (this.options.infinite) {
				this.slider._infinite.init().apply();
			}

			this.slider._setDimensions();
			this.slider._transitioner.init().apply(true, this.slider._setHeight.bind(this.slider));
			this.slider._setClasses();

			this.slider._navigation.refresh();
			this.slider._pagination.refresh();
		}
	}, {
		key: onResize,
		value: function value(e) {
			var newBreakPoint = this._getActiveBreakpoint();
			if (newBreakPoint.slidesToShow !== this._currentBreakpoint.slidesToShow) {
				this._currentBreakpoint = newBreakPoint;
				this.apply();
			}
		}
	}]);

	return Breakpoints;
}();

/* harmony default export */ __webpack_exports__["a"] = (Breakpoints);

/***/ }),
/* 10 */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _toConsumableArray(arr) { if (Array.isArray(arr)) { for (var i = 0, arr2 = Array(arr.length); i < arr.length; i++) { arr2[i] = arr[i]; } return arr2; } else { return Array.from(arr); } }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var Infinite = function () {
	function Infinite(slider) {
		_classCallCheck(this, Infinite);

		this.slider = slider;
	}

	_createClass(Infinite, [{
		key: 'init',
		value: function init() {
			if (this.slider.options.infinite && this.slider.options.effect === 'translate') {
				if (this.slider.options.centerMode) {
					this._infiniteCount = Math.ceil(this.slider.slidesToShow + this.slider.slidesToShow / 2);
				} else {
					this._infiniteCount = this.slider.slidesToShow;
				}

				var frontClones = [];
				var slideIndex = 0;
				for (var i = this.slider.state.length; i > this.slider.state.length - 1 - this._infiniteCount; i -= 1) {
					slideIndex = i - 1;
					frontClones.unshift(this._cloneSlide(this.slider.slides[slideIndex], slideIndex - this.slider.state.length));
				}

				var backClones = [];
				for (var _i = 0; _i < this._infiniteCount + this.slider.state.length; _i += 1) {
					backClones.push(this._cloneSlide(this.slider.slides[_i % this.slider.state.length], _i + this.slider.state.length));
				}

				this.slider.slides = [].concat(frontClones, _toConsumableArray(this.slider.slides), backClones);
			}
			return this;
		}
	}, {
		key: 'apply',
		value: function apply() {}
	}, {
		key: 'onTransitionEnd',
		value: function onTransitionEnd(e) {
			if (this.slider.options.infinite) {
				if (this.slider.state.next >= this.slider.state.length) {
					this.slider.state.index = this.slider.state.next = this.slider.state.next - this.slider.state.length;
					this.slider.transitioner.apply(true);
				} else if (this.slider.state.next < 0) {
					this.slider.state.index = this.slider.state.next = this.slider.state.length + this.slider.state.next;
					this.slider.transitioner.apply(true);
				}
			}
		}
	}, {
		key: '_cloneSlide',
		value: function _cloneSlide(slide, index) {
			var newSlide = slide.cloneNode(true);
			newSlide.dataset.sliderIndex = index;
			newSlide.dataset.cloned = true;
			var ids = newSlide.querySelectorAll('[id]') || [];
			ids.forEach(function (id) {
				id.setAttribute('id', '');
			});
			return newSlide;
		}
	}]);

	return Infinite;
}();

/* harmony default export */ __webpack_exports__["a"] = (Infinite);

/***/ }),
/* 11 */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_0__utils_dom__ = __webpack_require__(12);
var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }



var Loop = function () {
	function Loop(slider) {
		_classCallCheck(this, Loop);

		this.slider = slider;
	}

	_createClass(Loop, [{
		key: "init",
		value: function init() {
			return this;
		}
	}, {
		key: "apply",
		value: function apply() {
			if (this.slider.options.loop) {
				if (this.slider.state.next > 0) {
					if (this.slider.state.next < this.slider.state.length) {
						if (this.slider.state.next > this.slider.state.length - this.slider.slidesToShow && Object(__WEBPACK_IMPORTED_MODULE_0__utils_dom__["a" /* isInViewport */])(this.slider._slides[this.slider.state.length - 1], this.slider.wrapper)) {
							this.slider.state.next = 0;
						} else {
							this.slider.state.next = Math.min(Math.max(this.slider.state.next, 0), this.slider.state.length - this.slider.slidesToShow);
						}
					} else {
						this.slider.state.next = 0;
					}
				} else {
					if (this.slider.state.next <= 0 - this.slider.slidesToScroll) {
						this.slider.state.next = this.slider.state.length - this.slider.slidesToShow;
					} else {
						this.slider.state.next = 0;
					}
				}
			}
		}
	}]);

	return Loop;
}();

/* harmony default export */ __webpack_exports__["a"] = (Loop);

/***/ }),
/* 12 */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "a", function() { return isInViewport; });
var isInViewport = function isInViewport(element, html) {
	var rect = element.getBoundingClientRect();
	html = html || document.documentElement;
	return rect.top >= 0 && rect.left >= 0 && rect.bottom <= (window.innerHeight || html.clientHeight) && rect.right <= (window.innerWidth || html.clientWidth);
};

/***/ }),
/* 13 */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_0__templates_navigation__ = __webpack_require__(14);
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_1__utils_detect_supportsPassive__ = __webpack_require__(1);
var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }




var Navigation = function () {
	function Navigation(slider) {
		_classCallCheck(this, Navigation);

		this.slider = slider;

		this._clickEvents = ['click', 'touch'];
		this._supportsPassive = Object(__WEBPACK_IMPORTED_MODULE_1__utils_detect_supportsPassive__["a" /* default */])();

		this.onPreviousClick = this.onPreviousClick.bind(this);
		this.onNextClick = this.onNextClick.bind(this);
		this.onKeyUp = this.onKeyUp.bind(this);
	}

	_createClass(Navigation, [{
		key: 'init',
		value: function init() {
			this.node = document.createRange().createContextualFragment(Object(__WEBPACK_IMPORTED_MODULE_0__templates_navigation__["a" /* default */])(this.slider.options.icons));
			this._ui = {
				previous: this.node.querySelector('.slider-navigation-previous'),
				next: this.node.querySelector('.slider-navigation-next')
			};

			this._unbindEvents();
			this._bindEvents();

			this.refresh();

			return this;
		}
	}, {
		key: 'destroy',
		value: function destroy() {
			this._unbindEvents();
		}
	}, {
		key: '_bindEvents',
		value: function _bindEvents() {
			var _this = this;

			this.slider.wrapper.addEventListener('keyup', this.onKeyUp);
			this._clickEvents.forEach(function (clickEvent) {
				_this._ui.previous.addEventListener(clickEvent, _this.onPreviousClick);
				_this._ui.next.addEventListener(clickEvent, _this.onNextClick);
			});
		}
	}, {
		key: '_unbindEvents',
		value: function _unbindEvents() {
			var _this2 = this;

			this.slider.wrapper.removeEventListener('keyup', this.onKeyUp);
			this._clickEvents.forEach(function (clickEvent) {
				_this2._ui.previous.removeEventListener(clickEvent, _this2.onPreviousClick);
				_this2._ui.next.removeEventListener(clickEvent, _this2.onNextClick);
			});
		}
	}, {
		key: 'onNextClick',
		value: function onNextClick(e) {
			if (!this._supportsPassive) {
				e.preventDefault();
			}

			if (this.slider.options.navigation) {
				this.slider.next();
			}
		}
	}, {
		key: 'onPreviousClick',
		value: function onPreviousClick(e) {
			if (!this._supportsPassive) {
				e.preventDefault();
			}

			if (this.slider.options.navigation) {
				this.slider.previous();
			}
		}
	}, {
		key: 'onKeyUp',
		value: function onKeyUp(e) {
			if (this.slider.options.keyNavigation) {
				if (e.key === 'ArrowRight' || e.key === 'Right') {
					this.slider.next();
				} else if (e.key === 'ArrowLeft' || e.key === 'Left') {
					this.slider.previous();
				}
			}
		}
	}, {
		key: 'refresh',
		value: function refresh() {
			// let centerOffset = Math.floor(this.options.slidesToShow / 2);
			if (!this.slider.options.loop && !this.slider.options.infinite) {
				if (this.slider.options.navigation && this.slider.state.length > this.slider.slidesToShow) {
					this._ui.previous.classList.remove('is-hidden');
					this._ui.next.classList.remove('is-hidden');
					if (this.slider.state.next === 0) {
						this._ui.previous.classList.add('is-hidden');
						this._ui.next.classList.remove('is-hidden');
					} else if (this.slider.state.next >= this.slider.state.length - this.slider.slidesToShow && !this.slider.options.centerMode) {
						this._ui.previous.classList.remove('is-hidden');
						this._ui.next.classList.add('is-hidden');
					} else if (this.slider.state.next >= this.slider.state.length - 1 && this.slider.options.centerMode) {
						this._ui.previous.classList.remove('is-hidden');
						this._ui.next.classList.add('is-hidden');
					}
				} else {
					this._ui.previous.classList.add('is-hidden');
					this._ui.next.classList.add('is-hidden');
				}
			}
		}
	}, {
		key: 'render',
		value: function render() {
			return this.node;
		}
	}]);

	return Navigation;
}();

/* harmony default export */ __webpack_exports__["a"] = (Navigation);

/***/ }),
/* 14 */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
/* harmony default export */ __webpack_exports__["a"] = (function (icons) {
	return "<div class=\"slider-navigation-previous\">" + icons.previous + "</div>\n<div class=\"slider-navigation-next\">" + icons.next + "</div>";
});

/***/ }),
/* 15 */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_0__templates_pagination__ = __webpack_require__(16);
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_1__templates_pagination_page__ = __webpack_require__(17);
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_2__utils_detect_supportsPassive__ = __webpack_require__(1);
var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }





var Pagination = function () {
	function Pagination(slider) {
		_classCallCheck(this, Pagination);

		this.slider = slider;

		this._clickEvents = ['click', 'touch'];
		this._supportsPassive = Object(__WEBPACK_IMPORTED_MODULE_2__utils_detect_supportsPassive__["a" /* default */])();

		this.onPageClick = this.onPageClick.bind(this);
		this.onResize = this.onResize.bind(this);
	}

	_createClass(Pagination, [{
		key: 'init',
		value: function init() {
			this._pages = [];
			this.node = document.createRange().createContextualFragment(Object(__WEBPACK_IMPORTED_MODULE_0__templates_pagination__["a" /* default */])());
			this._ui = {
				container: this.node.firstChild
			};

			this._count = Math.ceil((this.slider.state.length - this.slider.slidesToShow) / this.slider.slidesToScroll);

			this._draw();
			this.refresh();

			return this;
		}
	}, {
		key: 'destroy',
		value: function destroy() {
			this._unbindEvents();
		}
	}, {
		key: '_bindEvents',
		value: function _bindEvents() {
			var _this = this;

			window.addEventListener('resize', this.onResize);
			window.addEventListener('orientationchange', this.onResize);

			this._clickEvents.forEach(function (clickEvent) {
				_this._pages.forEach(function (page) {
					return page.addEventListener(clickEvent, _this.onPageClick);
				});
			});
		}
	}, {
		key: '_unbindEvents',
		value: function _unbindEvents() {
			var _this2 = this;

			window.removeEventListener('resize', this.onResize);
			window.removeEventListener('orientationchange', this.onResize);

			this._clickEvents.forEach(function (clickEvent) {
				_this2._pages.forEach(function (page) {
					return page.removeEventListener(clickEvent, _this2.onPageClick);
				});
			});
		}
	}, {
		key: '_draw',
		value: function _draw() {
			this._ui.container.innerHTML = '';
			if (this.slider.options.pagination && this.slider.state.length > this.slider.slidesToShow) {
				for (var i = 0; i <= this._count; i++) {
					var newPageNode = document.createRange().createContextualFragment(Object(__WEBPACK_IMPORTED_MODULE_1__templates_pagination_page__["a" /* default */])()).firstChild;
					newPageNode.dataset.index = i * this.slider.slidesToScroll;
					this._pages.push(newPageNode);
					this._ui.container.appendChild(newPageNode);
				}
				this._bindEvents();
			}
		}
	}, {
		key: 'onPageClick',
		value: function onPageClick(e) {
			if (!this._supportsPassive) {
				e.preventDefault();
			}

			this.slider.state.next = e.currentTarget.dataset.index;
			this.slider.show();
		}
	}, {
		key: 'onResize',
		value: function onResize() {
			this._draw();
		}
	}, {
		key: 'refresh',
		value: function refresh() {
			var _this3 = this;

			var newCount = void 0;

			if (this.slider.options.infinite) {
				newCount = Math.ceil(this.slider.state.length - 1 / this.slider.slidesToScroll);
			} else {
				newCount = Math.ceil((this.slider.state.length - this.slider.slidesToShow) / this.slider.slidesToScroll);
			}
			if (newCount !== this._count) {
				this._count = newCount;
				this._draw();
			}

			this._pages.forEach(function (page) {
				page.classList.remove('is-active');
				if (parseInt(page.dataset.index, 10) === _this3.slider.state.next % _this3.slider.state.length) {
					page.classList.add('is-active');
				}
			});
		}
	}, {
		key: 'render',
		value: function render() {
			return this.node;
		}
	}]);

	return Pagination;
}();

/* harmony default export */ __webpack_exports__["a"] = (Pagination);

/***/ }),
/* 16 */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
/* harmony default export */ __webpack_exports__["a"] = (function () {
	return "<div class=\"slider-pagination\"></div>";
});

/***/ }),
/* 17 */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
/* harmony default export */ __webpack_exports__["a"] = (function () {
  return "<div class=\"slider-page\"></div>";
});

/***/ }),
/* 18 */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_0__utils_coordinate__ = __webpack_require__(4);
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_1__utils_detect_supportsPassive__ = __webpack_require__(1);
var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }




var Swipe = function () {
	function Swipe(slider) {
		_classCallCheck(this, Swipe);

		this.slider = slider;

		this._supportsPassive = Object(__WEBPACK_IMPORTED_MODULE_1__utils_detect_supportsPassive__["a" /* default */])();

		this.onStartDrag = this.onStartDrag.bind(this);
		this.onMoveDrag = this.onMoveDrag.bind(this);
		this.onStopDrag = this.onStopDrag.bind(this);

		this._init();
	}

	_createClass(Swipe, [{
		key: '_init',
		value: function _init() {}
	}, {
		key: 'bindEvents',
		value: function bindEvents() {
			var _this = this;

			this.slider.container.addEventListener('dragstart', function (e) {
				if (!_this._supportsPassive) {
					e.preventDefault();
				}
			});
			this.slider.container.addEventListener('mousedown', this.onStartDrag);
			this.slider.container.addEventListener('touchstart', this.onStartDrag);

			window.addEventListener('mousemove', this.onMoveDrag);
			window.addEventListener('touchmove', this.onMoveDrag);

			window.addEventListener('mouseup', this.onStopDrag);
			window.addEventListener('touchend', this.onStopDrag);
			window.addEventListener('touchcancel', this.onStopDrag);
		}
	}, {
		key: 'unbindEvents',
		value: function unbindEvents() {
			var _this2 = this;

			this.slider.container.removeEventListener('dragstart', function (e) {
				if (!_this2._supportsPassive) {
					e.preventDefault();
				}
			});
			this.slider.container.removeEventListener('mousedown', this.onStartDrag);
			this.slider.container.removeEventListener('touchstart', this.onStartDrag);

			window.removeEventListener('mousemove', this.onMoveDrag);
			window.removeEventListener('touchmove', this.onMoveDrag);

			window.removeEventListener('mouseup', this.onStopDrag);
			window.removeEventListener('mouseup', this.onStopDrag);
			window.removeEventListener('touchcancel', this.onStopDrag);
		}

		/**
   * @param {MouseEvent|TouchEvent}
   */

	}, {
		key: 'onStartDrag',
		value: function onStartDrag(e) {
			if (e.touches) {
				if (e.touches.length > 1) {
					return;
				} else {
					e = e.touches[0];
				}
			}

			this._origin = new __WEBPACK_IMPORTED_MODULE_0__utils_coordinate__["a" /* default */](e.screenX, e.screenY);
			this.width = this.slider.wrapperWidth;
			this.slider.transitioner.disable();
		}

		/**
   * @param {MouseEvent|TouchEvent}
   */

	}, {
		key: 'onMoveDrag',
		value: function onMoveDrag(e) {
			if (this._origin) {
				var point = e.touches ? e.touches[0] : e;
				this._lastTranslate = new __WEBPACK_IMPORTED_MODULE_0__utils_coordinate__["a" /* default */](point.screenX - this._origin.x, point.screenY - this._origin.y);
				if (e.touches) {
					if (Math.abs(this._lastTranslate.x) > Math.abs(this._lastTranslate.y)) {
						if (!this._supportsPassive) {
							e.preventDefault();
						}
						e.stopPropagation();
					}
				}
			}
		}

		/**
   * @param {MouseEvent|TouchEvent}
   */

	}, {
		key: 'onStopDrag',
		value: function onStopDrag(e) {
			if (this._origin && this._lastTranslate) {
				if (Math.abs(this._lastTranslate.x) > 0.2 * this.width) {
					if (this._lastTranslate.x < 0) {
						this.slider.next();
					} else {
						this.slider.previous();
					}
				} else {
					this.slider.show(true);
				}
			}
			this._origin = null;
			this._lastTranslate = null;
		}
	}]);

	return Swipe;
}();

/* harmony default export */ __webpack_exports__["a"] = (Swipe);

/***/ }),
/* 19 */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_0__transitions_fade__ = __webpack_require__(20);
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_1__transitions_translate__ = __webpack_require__(21);
var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }




var Transitioner = function () {
	function Transitioner(slider) {
		_classCallCheck(this, Transitioner);

		this.slider = slider;
		this.options = slider.options;

		this._animating = false;
		this._animation = undefined;

		this._translate = new __WEBPACK_IMPORTED_MODULE_1__transitions_translate__["a" /* default */](this, slider, slider.options);
		this._fade = new __WEBPACK_IMPORTED_MODULE_0__transitions_fade__["a" /* default */](this, slider, slider.options);
	}

	_createClass(Transitioner, [{
		key: 'init',
		value: function init() {
			this._fade.init();
			this._translate.init();
			return this;
		}
	}, {
		key: 'isAnimating',
		value: function isAnimating() {
			return this._animating;
		}
	}, {
		key: 'enable',
		value: function enable() {
			this._animation && this._animation.enable();
		}
	}, {
		key: 'disable',
		value: function disable() {
			this._animation && this._animation.disable();
		}
	}, {
		key: 'apply',
		value: function apply(force, callback) {
			// If we don't force refresh and animation in progress then return
			if (this._animating && !force) {
				return;
			}

			switch (this.options.effect) {
				case 'fade':
					this._animation = this._fade;
					break;
				case 'translate':
				default:
					this._animation = this._translate;
					break;
			}

			this._animationCallback = callback;

			if (force) {
				this._animation && this._animation.disable();
			} else {
				this._animation && this._animation.enable();
				this._animating = true;
			}

			this._animation && this._animation.apply();

			if (force) {
				this.end();
			}
		}
	}, {
		key: 'end',
		value: function end() {
			this._animating = false;
			this._animation = undefined;
			this.slider.state.index = this.slider.state.next;
			if (this._animationCallback) {
				this._animationCallback();
			}
		}
	}]);

	return Transitioner;
}();

/* harmony default export */ __webpack_exports__["a"] = (Transitioner);

/***/ }),
/* 20 */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_0__utils_css__ = __webpack_require__(0);
var _extends = Object.assign || function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; };

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }



var Fade = function () {
	function Fade(transitioner, slider) {
		var options = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : {};

		_classCallCheck(this, Fade);

		this.transitioner = transitioner;
		this.slider = slider;
		this.options = _extends({}, options);
	}

	_createClass(Fade, [{
		key: 'init',
		value: function init() {
			var _this = this;

			if (this.options.effect === 'fade') {
				this.slider.slides.forEach(function (slide, index) {
					Object(__WEBPACK_IMPORTED_MODULE_0__utils_css__["a" /* css */])(slide, {
						position: 'absolute',
						left: 0,
						top: 0,
						bottom: 0,
						'z-index': slide.dataset.sliderIndex == _this.slider.state.index ? 0 : -2,
						opacity: slide.dataset.sliderIndex == _this.slider.state.index ? 1 : 0
					});
				});
			}
			return this;
		}
	}, {
		key: 'enable',
		value: function enable() {
			var _this2 = this;

			this._oldSlide = this.slider.slides.filter(function (slide) {
				return slide.dataset.sliderIndex == _this2.slider.state.index;
			})[0];
			this._newSlide = this.slider.slides.filter(function (slide) {
				return slide.dataset.sliderIndex == _this2.slider.state.next;
			})[0];
			if (this._newSlide) {
				this._newSlide.addEventListener('transitionend', this.onTransitionEnd.bind(this));
				this._newSlide.style.transition = this.options.duration + 'ms ' + this.options.timing;
				if (this._oldSlide) {
					this._oldSlide.addEventListener('transitionend', this.onTransitionEnd.bind(this));
					this._oldSlide.style.transition = this.options.duration + 'ms ' + this.options.timing;
				}
			}
		}
	}, {
		key: 'disable',
		value: function disable() {
			var _this3 = this;

			this._oldSlide = this.slider.slides.filter(function (slide) {
				return slide.dataset.sliderIndex == _this3.slider.state.index;
			})[0];
			this._newSlide = this.slider.slides.filter(function (slide) {
				return slide.dataset.sliderIndex == _this3.slider.state.next;
			})[0];
			if (this._newSlide) {
				this._newSlide.removeEventListener('transitionend', this.onTransitionEnd.bind(this));
				this._newSlide.style.transition = 'none';
				if (this._oldSlide) {
					this._oldSlide.removeEventListener('transitionend', this.onTransitionEnd.bind(this));
					this._oldSlide.style.transition = 'none';
				}
			}
		}
	}, {
		key: 'apply',
		value: function apply(force) {
			var _this4 = this;

			this._oldSlide = this.slider.slides.filter(function (slide) {
				return slide.dataset.sliderIndex == _this4.slider.state.index;
			})[0];
			this._newSlide = this.slider.slides.filter(function (slide) {
				return slide.dataset.sliderIndex == _this4.slider.state.next;
			})[0];

			if (this._oldSlide && this._newSlide) {
				Object(__WEBPACK_IMPORTED_MODULE_0__utils_css__["a" /* css */])(this._oldSlide, {
					opacity: 0
				});
				Object(__WEBPACK_IMPORTED_MODULE_0__utils_css__["a" /* css */])(this._newSlide, {
					opacity: 1,
					'z-index': force ? 0 : -1
				});
			}
		}
	}, {
		key: 'onTransitionEnd',
		value: function onTransitionEnd(e) {
			if (this.options.effect === 'fade') {
				if (this.transitioner.isAnimating() && e.target == this._newSlide) {
					if (this._newSlide) {
						Object(__WEBPACK_IMPORTED_MODULE_0__utils_css__["a" /* css */])(this._newSlide, {
							'z-index': 0
						});
						this._newSlide.removeEventListener('transitionend', this.onTransitionEnd.bind(this));
					}
					if (this._oldSlide) {
						Object(__WEBPACK_IMPORTED_MODULE_0__utils_css__["a" /* css */])(this._oldSlide, {
							'z-index': -2
						});
						this._oldSlide.removeEventListener('transitionend', this.onTransitionEnd.bind(this));
					}
				}
				this.transitioner.end();
			}
		}
	}]);

	return Fade;
}();

/* harmony default export */ __webpack_exports__["a"] = (Fade);

/***/ }),
/* 21 */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_0__utils_coordinate__ = __webpack_require__(4);
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_1__utils_css__ = __webpack_require__(0);
var _extends = Object.assign || function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; };

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }




var Translate = function () {
	function Translate(transitioner, slider) {
		var options = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : {};

		_classCallCheck(this, Translate);

		this.transitioner = transitioner;
		this.slider = slider;
		this.options = _extends({}, options);

		this.onTransitionEnd = this.onTransitionEnd.bind(this);
	}

	_createClass(Translate, [{
		key: 'init',
		value: function init() {
			this._position = new __WEBPACK_IMPORTED_MODULE_0__utils_coordinate__["a" /* default */](this.slider.container.offsetLeft, this.slider.container.offsetTop);
			this._bindEvents();
			return this;
		}
	}, {
		key: 'destroy',
		value: function destroy() {
			this._unbindEvents();
		}
	}, {
		key: '_bindEvents',
		value: function _bindEvents() {
			this.slider.container.addEventListener('transitionend', this.onTransitionEnd);
		}
	}, {
		key: '_unbindEvents',
		value: function _unbindEvents() {
			this.slider.container.removeEventListener('transitionend', this.onTransitionEnd);
		}
	}, {
		key: 'enable',
		value: function enable() {
			this.slider.container.style.transition = this.options.duration + 'ms ' + this.options.timing;
		}
	}, {
		key: 'disable',
		value: function disable() {
			this.slider.container.style.transition = 'none';
		}
	}, {
		key: 'apply',
		value: function apply() {
			var _this = this;

			var maxOffset = void 0;
			if (this.options.effect === 'translate') {
				var slide = this.slider.slides.filter(function (slide) {
					return slide.dataset.sliderIndex == _this.slider.state.next;
				})[0];
				var slideOffset = new __WEBPACK_IMPORTED_MODULE_0__utils_coordinate__["a" /* default */](slide.offsetLeft, slide.offsetTop);
				if (this.options.centerMode) {
					maxOffset = new __WEBPACK_IMPORTED_MODULE_0__utils_coordinate__["a" /* default */](Math.round(Object(__WEBPACK_IMPORTED_MODULE_1__utils_css__["e" /* width */])(this.slider.container)), Math.round(Object(__WEBPACK_IMPORTED_MODULE_1__utils_css__["b" /* height */])(this.slider.container)));
				} else {
					maxOffset = new __WEBPACK_IMPORTED_MODULE_0__utils_coordinate__["a" /* default */](Math.round(Object(__WEBPACK_IMPORTED_MODULE_1__utils_css__["e" /* width */])(this.slider.container) - Object(__WEBPACK_IMPORTED_MODULE_1__utils_css__["e" /* width */])(this.slider.wrapper)), Math.round(Object(__WEBPACK_IMPORTED_MODULE_1__utils_css__["b" /* height */])(this.slider.container) - Object(__WEBPACK_IMPORTED_MODULE_1__utils_css__["b" /* height */])(this.slider.wrapper)));
				}
				var nextOffset = new __WEBPACK_IMPORTED_MODULE_0__utils_coordinate__["a" /* default */](Math.min(Math.max(slideOffset.x * -1, maxOffset.x * -1), 0), Math.min(Math.max(slideOffset.y * -1, maxOffset.y * -1), 0));
				if (this.options.loop) {
					if (!this.options.vertical && Math.abs(this._position.x) > maxOffset.x) {
						nextOffset.x = 0;
						this.slider.state.next = 0;
					} else if (this.options.vertical && Math.abs(this._position.y) > maxOffset.y) {
						nextOffset.y = 0;
						this.slider.state.next = 0;
					}
				}

				this._position.x = nextOffset.x;
				this._position.y = nextOffset.y;
				if (this.options.centerMode) {
					this._position.x = this._position.x + this.slider.wrapperWidth / 2 - Object(__WEBPACK_IMPORTED_MODULE_1__utils_css__["e" /* width */])(slide) / 2;
				}

				if (this.slider.direction === 'rtl') {
					this._position.x = -this._position.x;
					this._position.y = -this._position.y;
				}
				this.slider.container.style.transform = 'translate3d(' + this._position.x + 'px, ' + this._position.y + 'px, 0)';

				/**
     * update the index with the nextIndex only if
     * the offset of the nextIndex is in the range of the maxOffset
     */
				if (slideOffset.x > maxOffset.x) {
					this.slider.transitioner.end();
				}
			}
		}
	}, {
		key: 'onTransitionEnd',
		value: function onTransitionEnd(e) {
			if (this.options.effect === 'translate') {

				if (this.transitioner.isAnimating() && e.target == this.slider.container) {
					if (this.options.infinite) {
						this.slider._infinite.onTransitionEnd(e);
					}
				}
				this.transitioner.end();
			}
		}
	}]);

	return Translate;
}();

/* harmony default export */ __webpack_exports__["a"] = (Translate);

/***/ }),
/* 22 */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
var defaultOptions = {
  initialSlide: 0,
  slidesToScroll: 1,
  slidesToShow: 1,

  navigation: true,
  navigationKeys: true,
  navigationSwipe: true,

  pagination: true,

  loop: false,
  infinite: false,

  effect: 'translate',
  duration: 300,
  timing: 'ease',

  autoplay: false,
  autoplaySpeed: 3000,
  pauseOnHover: true,
  breakpoints: [{
    changePoint: 480,
    slidesToShow: 1,
    slidesToScroll: 1
  }, {
    changePoint: 640,
    slidesToShow: 2,
    slidesToScroll: 2
  }, {
    changePoint: 768,
    slidesToShow: 3,
    slidesToScroll: 3
  }],

  onReady: null,
  icons: {
    'previous': '<svg viewBox="0 0 50 80" xml:space="preserve">\n      <polyline fill="currentColor" stroke-width=".5em" stroke-linecap="round" stroke-linejoin="round" points="45.63,75.8 0.375,38.087 45.63,0.375 "/>\n    </svg>',
    'next': '<svg viewBox="0 0 50 80" xml:space="preserve">\n      <polyline fill="currentColor" stroke-width=".5em" stroke-linecap="round" stroke-linejoin="round" points="0.375,0.375 45.63,38.087 0.375,75.8 "/>\n    </svg>'
  }
};

/* harmony default export */ __webpack_exports__["a"] = (defaultOptions);

/***/ }),
/* 23 */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
/* harmony default export */ __webpack_exports__["a"] = (function (id) {
  return "<div id=\"" + id + "\" class=\"slider\" tabindex=\"0\">\n    <div class=\"slider-container\"></div>\n  </div>";
});

/***/ }),
/* 24 */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
/* harmony default export */ __webpack_exports__["a"] = (function () {
  return "<div class=\"slider-item\"></div>";
});

/***/ })
/******/ ])["default"];
});