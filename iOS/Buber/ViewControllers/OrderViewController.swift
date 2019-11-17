//
//  OrderViewController.swift
//  Buber
//
//  Created by Alexey Salangin on 11/16/19.
//  Copyright © 2019 Alexey Salangin. All rights reserved.
//

import UIKit
import SnapKit

protocol OrderViewControllerDelegate: AnyObject {
    func orderViewControllerDidTapOrderButton(_ vc: OrderViewController)
    func orderViewControllerDidTapInBusButton(_ vc: OrderViewController)
    func orderViewControllerDidTapCancelButton(_ vc: OrderViewController)
}

final class OrderViewController: UIViewController {
    weak var delegate: OrderViewControllerDelegate?

    private let contentStackView = UIStackView()

    private let infoView = UIView()
    private let separatorView: UIView
    private let paymentInfoView = PaymentView()
    private let orderButton: UIButton
    private let inBusButton: UIButton
    private let cancelButton: UIButton

    init() {
        self.orderButton = Self.makeOrderButton()
        self.inBusButton = Self.makeInBusButton()
        self.cancelButton = Self.makeCancelButton()
        self.separatorView = Self.makeSeparatorView()

        super.init(nibName: nil, bundle: nil)

        self.view.backgroundColor = .white
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func setupForOrderState() {
        self.orderButton.isHidden = false
        self.paymentInfoView.isHidden = false
        self.cancelButton.isHidden = true
        self.inBusButton.isHidden = true
        self.contentStackView.setCustomSpacing(24, after: self.inBusButton)
        self.contentStackView.setCustomSpacing(24, after: self.separatorView)
    }

    func setupForWaitingState() {
        self.orderButton.isHidden = true
        self.paymentInfoView.isHidden = true
        self.cancelButton.isHidden = false
        self.inBusButton.isHidden = false
        self.contentStackView.setCustomSpacing(12, after: self.inBusButton)
        self.contentStackView.setCustomSpacing(18, after: self.separatorView)
    }

    override func viewDidLoad() {
        super.viewDidLoad()

        self.view.addSubview(self.contentStackView)
        self.contentStackView.snp.makeConstraints { make in
            make.top.equalTo(self.view.snp.top).offset(55)
            make.left.equalTo(self.view.snp.left).offset(24)
            make.right.equalTo(self.view.snp.right).offset(-24)
        }

        self.setupSubviews()
    }

    private func setupSubviews() {
        self.contentStackView.axis = .vertical
        self.contentStackView.spacing = 24

        self.contentStackView.addArrangedSubview(self.infoView)
        self.contentStackView.addArrangedSubview(self.separatorView)

        self.contentStackView.addArrangedSubview(self.paymentInfoView)
        self.contentStackView.addArrangedSubview(self.orderButton)

        self.contentStackView.addArrangedSubview(self.inBusButton)
        self.contentStackView.addArrangedSubview(self.cancelButton)

        self.setupInfoView()

        self.setupForOrderState()

        self.orderButton.addTarget(self, action: #selector(self.didTapOrderButton), for: .touchUpInside)
        self.inBusButton.addTarget(self, action: #selector(self.didTapInBusButton), for: .touchUpInside)
        self.cancelButton.addTarget(self, action: #selector(self.didTapCancelButton), for: .touchUpInside)
    }

    private func setupInfoView() {
        let busInfoView = BusInfoView()
        let routeInfoView = RouteInfoView()

        self.infoView.addSubview(busInfoView)
        self.infoView.addSubview(routeInfoView)

        busInfoView.snp.makeConstraints { make in
            make.top.left.bottom.equalToSuperview()
        }

        routeInfoView.snp.makeConstraints { make in
            make.top.right.bottom.equalToSuperview()
            make.left.equalTo(busInfoView.snp.right)
            make.width.equalTo(busInfoView)
        }
    }

    @objc
    private func didTapOrderButton() {
        self.delegate?.orderViewControllerDidTapOrderButton(self)
    }

    @objc
    private func didTapInBusButton() {
        self.delegate?.orderViewControllerDidTapInBusButton(self)
    }

    @objc
    private func didTapCancelButton() {
        self.delegate?.orderViewControllerDidTapCancelButton(self)
    }

    private static func makeSeparatorView() -> UIView {
        let separator = UIView()
        separator.backgroundColor = UIColor(red: 0.769, green: 0.769, blue: 0.769, alpha: 1)
        separator.snp.makeConstraints { make in
            make.height.equalTo(1)
        }
        return separator
    }

    private static func makeOrderButton() -> UIButton {
        let button = UIButton()
        button.setBackgroundImage(UIImage.imageWith(color: .app), for: .normal)
        button.layer.cornerRadius = 10
        button.layer.masksToBounds = true
        let attributedTitle = NSAttributedString(
            string: "Order bus",
            attributes: [
                .foregroundColor: UIColor.white,
                .font: UIFont.appMedium(16)
            ]
        )
        button.setAttributedTitle(attributedTitle, for: .normal)
        button.snp.makeConstraints { make in
            make.height.equalTo(46)
        }
        return button
    }

    private static func makeInBusButton() -> UIButton {
        let button = UIButton()
        button.setBackgroundImage(UIImage.imageWith(color: .app), for: .normal)
        button.layer.cornerRadius = 10
        button.layer.masksToBounds = true
        let attributedTitle = NSAttributedString(
            string: "I'm in the bus",
            attributes: [
                .foregroundColor: UIColor.white,
                .font: UIFont.appMedium(16)
            ]
        )
        button.setAttributedTitle(attributedTitle, for: .normal)
        button.snp.makeConstraints { make in
            make.height.equalTo(46)
        }
        return button
    }

    private static func makeCancelButton() -> UIButton {
        let button = UIButton()
        button.setBackgroundImage(UIImage.imageWith(color: .cancel), for: .normal)
        button.layer.cornerRadius = 10
        button.layer.masksToBounds = true
        let attributedTitle = NSAttributedString(
            string: "Cancel",
            attributes: [
                .foregroundColor: UIColor.white,
                .font: UIFont.appMedium(16)
            ]
        )
        button.setAttributedTitle(attributedTitle, for: .normal)
        button.snp.makeConstraints { make in
            make.height.equalTo(46)
        }
        return button
    }
}
